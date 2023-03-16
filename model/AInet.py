import torch
from torch import nn 
import pdb

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.ReLU(),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class AInet(nn.Module):
    def __init__(self):
        super(AInet, self).__init__()
        self.faceid_encoder = nn.ModuleList([
            Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # b,16,128    7
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # b,32,64      6
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # b,64,32    5
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # b,128,16   4
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # b,256,8    3 
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # b,512,4     2
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1)    # b,512,4   1 
            ])
        
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 16, 8
            Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  # 32, 8
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 64, 4 
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 128, 4
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1))  # 512, 4
        
        self.lstm = nn.LSTM(336, 256, 3, batch_first=True)
        
        self.faceid_decoder= nn.ModuleList([
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),    # b,512,4    1  
            Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=1, output_padding=0),  # b,512,4    2
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # b,512,8    3
            Conv2dTranspose(768, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # b,512,16    4
            Conv2dTranspose(384, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # b,512,32     5
            Conv2dTranspose(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # b,512,64       6
            Conv2dTranspose(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # b,512,64      7
        ])
        
        self.out_block = nn.Sequential(Conv2dTranspose(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       Conv2dTranspose(16, 3, kernel_size=3, stride=1, padding=1))
        
    def forward(self, face_id, audio):
        ## in_faceid b,3,256,256
        ## in_audio b,t,28,12
        b,t = audio.shape[0], audio.shape[1]
        lstm_in = audio.reshape(b, t, -1)
        
        hidden = (torch.autograd.Variable(torch.zeros(3, b, 256).cuda()),
                torch.autograd.Variable(torch.zeros(3, b, 256).cuda()))
        
        lstm_out, _ = self.lstm(lstm_in, hidden)   # b,t,256
        lstm_out = lstm_out.reshape(b, t, 16, 16)
        
        lstm_f = []
        for i in range(t):
            lstm_s = lstm_out[:,i,:,:].unsqueeze(1)
            lstm_s = self.audio_encoder(lstm_s)
            lstm_f.append(lstm_s)
            
        lstm_f = torch.stack(lstm_f, dim=1)   # b,t,512,4,4
        audio_block = torch.cat([lstm_f[:, i] for i in range(lstm_f.size(1))], dim=0)   # b*t,512,4,4
        face_id = torch.cat([face_id]* lstm_f.size(1), dim=0)
        
        feats = []
        x = face_id
        for f in self.faceid_encoder:
            x = f(x)
            feats.append(x)

        x = audio_block
        for f in self.faceid_decoder:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()
            
        out = self.out_block(x)
        out = out.reshape(b, 5, 3, 256, 256)
        
        return out 
    
            
        
        
        