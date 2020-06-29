import functools
import torch
import torch.nn as nn


def deconv_2d(in_channels,out_channels):

    deconv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    return nn.Sequential(*[deconv])

class Blocks(nn.Module):
    def __init__(self,res_block_num,conv_channels):
        super(Blocks,self).__init__()
        self.res_block_num=res_block_num
        for res_block_idx in range(self.res_block_num):
            conv_layer_1=nn.Conv2d(conv_channels,conv_channels,kernel_size=3,padding=1,stride=1)
            conv_layer_2=nn.Conv2d(conv_channels,conv_channels,kernel_size=3,padding=1,stride=1)
            self.add_module('%d'%(2*res_block_idx),nn.Sequential(conv_layer_1))
            self.add_module('%d'%(2*res_block_idx+1),nn.Sequential(conv_layer_2))

    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError('index %d is out of range'%(index))

        return(self._modules[str(index)])

    def __len__(self):
        return self.res_block_num 

class SeeMotionInDarkNet(nn.Module):
    def __init__(self,input_channels=3,base_dim=32,res_block_num=16):
        super(SeeMotionInDarkNet,self).__init__()
        

        self.res_block_num = res_block_num

        self.conv_1_1=nn.Conv2d(3,base_dim,kernel_size=3,padding=1,stride=1)
        self.conv_1_2=nn.Conv2d(base_dim,base_dim,kernel_size=3,padding=1,stride=1)

        self.conv_2_1=nn.Conv2d(base_dim,base_dim*2,kernel_size=3,padding=1,stride=1)
        self.conv_2_2=nn.Conv2d(base_dim*2,base_dim*2,kernel_size=3,padding=1,stride=1)

        self.conv_3_1=nn.Conv2d(base_dim*2,base_dim*4,kernel_size=3,padding=1,stride=1)
        self.conv_3_2=nn.Conv2d(base_dim*4,base_dim*4,kernel_size=3,padding=1,stride=1)

        self.conv_4_1=nn.Conv2d(base_dim*4,base_dim*8,kernel_size=3,padding=1,stride=1)
        self.conv_4_2=nn.Conv2d(base_dim*8,base_dim*8,kernel_size=3,padding=1,stride=1)

        self.conv_5_1=nn.Conv2d(base_dim*8,base_dim*16,kernel_size=3,padding=1,stride=1)
        self.conv_5_2=nn.Conv2d(base_dim*16,base_dim*16,kernel_size=3,padding=1,stride=1)

        self.res_block_list=Blocks(self.res_block_num,base_dim*16)
                 
        self.conv_after_res_block=nn.Conv2d(base_dim*16,base_dim*16,kernel_size=3,padding=1,stride=1)

        self.deconv_6=deconv_2d(base_dim*16,base_dim*8)
        self.conv_6_1=nn.Conv2d(base_dim*16,base_dim*8,kernel_size=3,padding=1,stride=1)
        self.conv_6_2=nn.Conv2d(base_dim*8,base_dim*8,kernel_size=3,padding=1,stride=1)

        self.deconv_7=deconv_2d(base_dim*8,base_dim*4)
        self.conv_7_1=nn.Conv2d(base_dim*8,base_dim*4,kernel_size=3,padding=1,stride=1)
        self.conv_7_2=nn.Conv2d(base_dim*4,base_dim*4,kernel_size=3,padding=1,stride=1)

        self.deconv_8=deconv_2d(base_dim*4,base_dim*2)
        self.conv_8_1=nn.Conv2d(base_dim*4,base_dim*2,kernel_size=3,padding=1,stride=1)
        self.conv_8_2=nn.Conv2d(base_dim*2,base_dim*2,kernel_size=3,padding=1,stride=1)

        self.deconv_9=deconv_2d(base_dim*2,base_dim)
        self.conv_9_1=nn.Conv2d(base_dim*2,base_dim,kernel_size=3,padding=1,stride=1)
        self.conv_9_2=nn.Conv2d(base_dim,base_dim,kernel_size=3,padding=1,stride=1)

        self.conv_10=nn.Conv2d(base_dim,3,kernel_size=3,padding=1,stride=1)


    def forward(self, input_tensor):

        self.pool=nn.MaxPool2d(2) 
        self.lrelu=nn.LeakyReLU(negative_slope=0.2)

        conv_1=self.lrelu(self.conv_1_1(input_tensor))
        conv_1=self.conv_1_2(conv_1)
        self.ln1 = nn.LayerNorm(conv_1.size()[1:], elementwise_affine=False)
        conv_1 = self.lrelu(self.ln1(conv_1))
        conv_1_pool=self.pool(conv_1)

        conv_2=self.conv_2_1(conv_1_pool)
        self.ln2_1 = nn.LayerNorm(conv_2.size()[1:], elementwise_affine=False)
        conv_2=self.conv_2_2(self.lrelu(self.ln2_1(conv_2)))
        self.ln2_2 = nn.LayerNorm(conv_2.size()[1:], elementwise_affine=False)
        conv_2=self.lrelu(self.ln2_2(conv_2))
        conv_2_pool=self.pool(conv_2)

        conv_3=self.conv_3_1(conv_2_pool)
        self.ln3_1 = nn.LayerNorm(conv_3.size()[1:], elementwise_affine=False)
        conv_3=self.conv_3_2(self.lrelu(self.ln3_1(conv_3)))
        self.ln3_2 = nn.LayerNorm(conv_3.size()[1:], elementwise_affine=False)
        conv_3=self.lrelu(self.ln3_2(conv_3))
        conv_3_pool=self.pool(conv_3)

        conv_4=self.conv_4_1(conv_3_pool)
        self.ln4_1 = nn.LayerNorm(conv_4.size()[1:], elementwise_affine=False)
        conv_4=self.conv_4_2(self.lrelu(self.ln4_1(conv_4)))
        self.ln4_2 = nn.LayerNorm(conv_4.size()[1:], elementwise_affine=False)
        conv_4=self.lrelu(self.ln4_2(conv_4))
        conv_4_pool=self.pool(conv_4)

        conv_5=self.conv_5_1(conv_4_pool)
        self.ln5_1 = nn.LayerNorm(conv_5.size()[1:], elementwise_affine=False)
        conv_5=self.conv_5_2(self.lrelu(self.ln5_1(conv_5)))
        self.ln5_2 = nn.LayerNorm(conv_5.size()[1:], elementwise_affine=False)
        conv_5=self.lrelu(self.ln5_2(conv_5))

        #res block
        conv_feature_end=conv_5
        for res_block_idx in range(self.res_block_num):
            conv_feature_begin = conv_feature_end
            conv_feature=self.res_block_list[2*res_block_idx](conv_feature_begin)
            self.ln_1 = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
            conv_feature = self.res_block_list[2*res_block_idx+1](self.lrelu(self.ln_1(conv_feature)))
            self.ln_2 = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
            conv_feature = self.ln_2(conv_feature)
            conv_feature_end = conv_feature_begin+conv_feature
            
        conv_feature=self.conv_after_res_block(conv_feature_end)
        self.ln = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
        conv_feature = self.ln(conv_feature)
        conv_feature=conv_feature+conv_5

        conv_6_up=self.deconv_6(conv_feature)
        conv_6=torch.cat((conv_6_up,conv_4),dim=1)
        conv_6=self.conv_6_1(conv_6)
        self.ln6_1 = nn.LayerNorm(conv_6.size()[1:], elementwise_affine=False)
        conv_6=self.conv_6_2(self.lrelu(self.ln6_1(conv_6)))
        self.ln6_2 = nn.LayerNorm(conv_6.size()[1:], elementwise_affine=False)
        conv_6=self.lrelu(self.ln6_2(conv_6))

        conv_7_up=self.deconv_7(conv_6)
        conv_7=torch.cat((conv_7_up,conv_3),dim=1)
        conv_7=self.conv_7_1(conv_7)
        self.ln7_1 = nn.LayerNorm(conv_7.size()[1:], elementwise_affine=False)
        conv_7=self.conv_7_2(self.lrelu(self.ln7_1(conv_7)))
        self.ln7_2 = nn.LayerNorm(conv_7.size()[1:], elementwise_affine=False)
        conv_7=self.lrelu(self.ln7_2(conv_7))

        conv_8_up=self.deconv_8(conv_7)
        conv_8=torch.cat((conv_8_up,conv_2),dim=1)
        conv_8=self.conv_8_1(conv_8)
        self.ln8_1 = nn.LayerNorm(conv_8.size()[1:], elementwise_affine=False)
        conv_8=self.conv_8_2(self.lrelu(self.ln8_1(conv_8)))
        self.ln8_2 = nn.LayerNorm(conv_8.size()[1:], elementwise_affine=False)
        conv_8=self.lrelu(self.ln8_2(conv_8))

        conv_9_up=self.deconv_9(conv_8)
        conv_9=torch.cat((conv_9_up,conv_1),dim=1)
        conv_9=self.conv_9_1(conv_9)
        self.ln9_1 = nn.LayerNorm(conv_9.size()[1:], elementwise_affine=False)
        conv_9=self.conv_9_2(self.lrelu(self.ln9_1(conv_9)))
        self.ln9_2 = nn.LayerNorm(conv_9.size()[1:], elementwise_affine=False)
        conv_9=self.lrelu(self.ln9_2(conv_9))

        out=self.conv_10(conv_9)


        return out
