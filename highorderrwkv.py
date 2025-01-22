import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from .refine import Refine
import torch.nn.functional as F
from torch.utils.cpp_extension import load
T_MAX = 512*512
torch.autograd.set_detect_anomaly(True)

wkv_cuda = load(name="wkv", sources=["rwkvcuda/wkv_op.cpp", "rwkvcuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())



class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out 



class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', 
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False) 
        
        self.recurrence = 2 
        
        self.omni_shift = OmniShift(dim=n_embd)


        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 


        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 



    def jit_func(self, x,kv_cache=None):
        # Mix x with the previous timestep to produce xk, xv, xr

        if kv_cache!=None:
            k = kv_cache[0]
            v=  kv_cache[1]
            k = self.key(k)
            v = self.value(v)
        else:
            k = self.key(x)
            v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v


        
        

    def forward(self, x,kv_cache=None):
        x,
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x,kv_cache) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        new_x = sr * x
        new_x = self.output(new_x)
        return new_x,[k,v]



class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd



        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omni_shift = OmniShift(dim=n_embd)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.evlo = nn.Linear(n_embd, n_embd, bias=False)


    def forward(self, x,channel_kv):


        if channel_kv!=None:
            kv = self.evlo(channel_kv)
        else:
            k = self.key(x)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x,kv

import random

class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id 
        self.repeat = 4

        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 


        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                   key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate,
                                   init_mode, key_norm=key_norm)



        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)


    def forward(self, ins): 
        # x = self.dwconv1(x) + x
        x,kv_cache,channel_kv = ins[0],ins[1],ins[2]
        kv_cache=None
        if not self.training:
            x = x.repeat(self.repeat,1,1)
            if kv_cache:
                kv_cache[0] = kv_cache[0].repeat(self.repeat,1,1)
                kv_cache[1] = kv_cache[1].repeat(self.repeat,1,1)
        B,N,C = x.shape
        N_index = list(range(0,N))
        N_shuffle = list(range(0,N))
        N_shuffle_list = []
        shuffle_x = x.clone()
        if kv_cache:
            shuffle_kv = [0,0]
            shuffle_kv[0] = kv_cache[0].clone()
            shuffle_kv[1] = kv_cache[1].clone()
        else:
            shuffle_kv=None
        for b in range(B):
            random.shuffle(N_shuffle)
            N_shuffle_list.append(N_shuffle)
            N_shuffle = list(range(0,N))
        for b,N_shuffle in enumerate(N_shuffle_list):
            shuffle_x[b,:,: ] = x[b,N_shuffle,:]
            if kv_cache:
                shuffle_kv[0][b,:,: ]= kv_cache[0][b,N_shuffle,:]
                shuffle_kv[1][b,:,: ]= kv_cache[1][b,N_shuffle,:]

        shuffle_x_out,shuffle_kv_out = self.att(self.ln1(shuffle_x),kv_cache=shuffle_kv) 
        RN_shuffle_list = []
        for b in range(B):
            RN_shuffle_list.append([N_shuffle_list[b][i] for  i in N_index])
        for b,rn_shuffle in enumerate(RN_shuffle_list):
            shuffle_x_out[b,rn_shuffle,:] = shuffle_x_out[b,N_index,:]
            shuffle_kv_out[0][b,rn_shuffle,:] = shuffle_kv_out[0][b,N_index,:]
            shuffle_kv_out[1][b,rn_shuffle,:] = shuffle_kv_out[1][b,N_index,:]

        out = shuffle_x_out
        kv_out = shuffle_kv_out
        if not self.training:
            out = torch.stack(out.chunk(self.repeat,dim=0),dim=0)
            out = torch.mean(out,dim=0)
            kv_out[0] = torch.stack(kv_out[0].chunk(self.repeat,dim=0),dim=0)
            kv_out[0] = torch.mean(kv_out[0],dim=0)
            kv_out[1] = torch.stack(kv_out[1].chunk(self.repeat,dim=0),dim=0)
            kv_out[1] = torch.mean(kv_out[1],dim=0)

        # out,kv_out =self.att(self.ln1(x),kv_cache=kv_cache) 
        x = x + self.gamma1 * out
        out_channel,channel_kv_out = self.ffn(self.ln2(x),channel_kv) 
        x = x + self.gamma2 * out_channel
        return [x,kv_out,channel_kv_out]
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Linear(dim, hidden_features)


        self.project_out = nn.Linear(hidden_features, dim)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2)
        x = F.fold(x,(128,128),(1,1),stride=1)
        return x
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels,out_channels):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        # 使用卷积层来提取patch，每个patch作    为一个token
        self.conv = nn.Conv2d(in_channels, in_channels * patch_size * patch_size, 
                              kernel_size=patch_size, 
                              stride=patch_size, 
                              padding=0)
        self.proj = nn.Linear(in_channels * patch_size * patch_size,out_channels)
    def forward(self, x):
        b, c, h, w = x.size()
        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Height and Width must be divisible by patch_size"
        
        # 使用卷积层提取patch
        x = self.conv(x)  # 形状变为 (b, c*patch_size*patch_size, h//patch_size, w//patch_size)
        x = x.view(b,c * self.patch_size * self.patch_size,-1)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x
class CrossRWKVBlock(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id 
         

        self.ln0 = nn.LayerNorm(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.ln3 = nn.LayerNorm(n_embd) 

        self.sa = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                   key_norm=key_norm)


        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate,
                                   init_mode, key_norm=key_norm)


        self.gamma0 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)


    def forward(self,x,y,kv_cache,channel_kv): 
        b,n,c = x.shape
        resi_x = x
        kv_cache=None
        if self.layer_id%2==0:
            resi_x[:,:,:c//2] = y[:,:,:c//2]
        else:
            resi_x[:,:,c//2:] = y[:,:,c//2:]
        if not self.training:
            resi_x = resi_x.repeat(self.repeat,1,1)
            if kv_cache:
                kv_cache[0] = kv_cache[0].repeat(self.repeat,1,1)
                kv_cache[1] = kv_cache[1].repeat(self.repeat,1,1)

        B,N,C = resi_x.shape
        N_index = list(range(0,N))
        N_shuffle = list(range(0,N))
        N_shuffle_list = []
        shuffle_x = resi_x.clone()
        if kv_cache:
            shuffle_kv=[0,0]
            shuffle_kv[0] = kv_cache[0].clone()
            shuffle_kv[1] = kv_cache[1].clone()

        else:
            shuffle_kv=None
        for b in range(B):
            random.shuffle(N_shuffle)
            N_shuffle_list.append(N_shuffle)
            N_shuffle = list(range(0,N))
        for b,N_shuffle in enumerate(N_shuffle_list):
            shuffle_x[b,:,: ] = resi_x[b,N_shuffle,:]
            if kv_cache:
                shuffle_kv[0][b,:,: ]= kv_cache[0][b,N_shuffle,:]
                shuffle_kv[1][b,:,: ]= kv_cache[1][b,N_shuffle,:]

        shuffle_x_out,shuffle_kv_out = self.sa(self.ln1(shuffle_x),kv_cache=shuffle_kv) 
        RN_shuffle_list = []
        for b in range(B):
            RN_shuffle_list.append([N_shuffle_list[b][i] for   i in N_index])
        for b,rn_shuffle in enumerate(RN_shuffle_list):
            shuffle_x_out[b,rn_shuffle,:] = shuffle_x_out[b,N_index,:]
            shuffle_kv_out[0][b,rn_shuffle,:] = shuffle_kv_out[0][b,N_index,:]
            shuffle_kv_out[1][b,rn_shuffle,:] = shuffle_kv_out[1][b,N_index,:]

        out = shuffle_x_out
        kv_out = shuffle_kv_out
        if not self.training:
            out = torch.stack(out.chunk(self.repeat,dim=0),dim=0)
            out = torch.mean(out,dim=0)
            kv_out[0] = torch.stack(kv_out[0].chunk(self.repeat,dim=0),dim=0)
            kv_out[0] = torch.mean(kv_out[0],dim=0)
            kv_out[1] = torch.stack(kv_out[1].chunk(self.repeat,dim=0),dim=0)
            kv_out[1] = torch.mean(kv_out[1],dim=0)

        x = x + self.gamma1 * out
        out,channel_kv_out = self.ffn(self.ln2(x),channel_kv) 
        x = x + self.gamma2 * out
        return x,kv_out,channel_kv_out
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi
class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,16,3,1,1),HinResBlock(16,16),HinResBlock(16,16),HinResBlock(16,16),HinResBlock(16,16))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,16,3,1,1),HinResBlock(16,16),HinResBlock(16,16),HinResBlock(16,16),HinResBlock(16,16))
        self.ms_feature_extraction = nn.Sequential(*[Block(base_filter,1,i) for i in range(8)])
        self.pan_feature_extraction =nn.Sequential(*[Block(base_filter,1,i) for i in range(8)])
        self.fuse1 = CrossRWKVBlock(base_filter,1,0)
        self.fuse2 = CrossRWKVBlock(base_filter,1,1)
        self.fuse3 = CrossRWKVBlock(base_filter,1,2)
        self.fuse4 = CrossRWKVBlock(base_filter,1,3)
        self.fuse5 = CrossRWKVBlock(base_filter,1,4)

        self.output = Refine(32,4)
        self.patchfy=PatchEmbed(self.patch_size,16,base_filter)
        self.unpatch = PatchUnEmbed(base_filter)
        self.proj =nn.Conv2d(16,4,1,1,0)
        self.simple_fusion = nn.Linear(base_filter*2,base_filter)
    def forward(self,ms,_,pan):

        ms_bic = F.interpolate(ms,scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        pan_f = self.pan_encoder(pan)
        b,c,h,w = ms_f.shape
        
        ms_f = self.patchfy(ms_f)
        pan_f = self.patchfy(pan_f)
        ms_f,cache_ms,cache_can_ms = self.ms_feature_extraction([ms_f,None,None])
        pan_f,cache_pan,cache_can_pan = self.pan_feature_extraction([pan_f,None,None])
        ms_f = self.simple_fusion(torch.concat([ms_f,pan_f],dim=2))
        pan_f = self.simple_fusion(torch.concat([pan_f,ms_f],dim=2))
        ms_f,cache,cancache = self.fuse1(ms_f,pan_f,None,None)
        ms_f,cache,cancache = self.fuse2(ms_f,pan_f,cache,cancache)
        ms_f,cache,cancache = self.fuse3(ms_f,pan_f,cache,cancache)
        ms_f,cache,cancache = self.fuse4(ms_f,pan_f,cache,cancache)
        ms_f,cache,cancache = self.fuse5(ms_f,pan_f,cache,cancache)

        ms_f = self.unpatch(ms_f,(128,128))
        hrms = self.output(ms_f)+ms_bic

        return hrms


