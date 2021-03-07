import numpy as np
import sys
import os
import soundfile as sf

import time

m = np.load(sys.argv[1])
if m.shape[0] == 20:
    m = m.T
assert m.shape[1] == 20

out_fea = np.zeros((m.shape[0], 55), dtype=np.float32)
out_fea[:, :18] = m[:, :18]
out_fea[:, 36:38] = m[:, 18:]
out_fea.tofile('test.f32')

start = time.time()
os.system("./lpcnet_demo -synthesis test.f32 out.pcm")
end = time.time()
print("time: ", round(end-start, 2))
a = np.fromfile('out.pcm', dtype=np.int16)
print("len(wav): ", str(round(len(a)/16000, 2)) + "s")
sf.write("out.wav", a, 16000, "PCM_16")
#os.system("ffmpeg -y -f s16le -ar 16k -ac 1 -i out.pcm out.wav")
os.system("rm out.pcm")
os.system("rm test.f32")

