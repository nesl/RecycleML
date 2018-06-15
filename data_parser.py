import numpy as np
import cv2
import glob
import os
import subprocess
import gzip
import csv

import librosa
import numpy as np
import subprocess
import gzip
import csv
import io

import pickle

from tqdm import tqdm

# video_parent_dir='/home/tianwei/Dist_sensorysub/0HotEdge_Data/Personalization_full'    # personal
video_parent_dir='/home/tianwei/Dist_sensorysub/0HotEdge_Data/Data-Train'    # train
# video_parent_dir='/home/tianwei/Dist_sensorysub/0HotEdge_Data/Data-Test'    # test
IMU_parent_dir='/home/tianwei/Dist_sensorysub/0HotEdge_Data/IMU_data/raw'

sub_video_dirs=['downstair','upstair','run','jump','walk', 'handwashing','exercise'] # all
# sub_video_dirs=['run','jump','walk', 'handwashing','exercise'] # personalize
imu_sensor_dirs=['raw3','raw4','raw11','raw12']

TimeWindow = 2
Data_window_watch=40
Data_window_video=45
TimeWindow_slide=0.4
img_size = 64

def parse_IMU_files(parent_dir, sub_dirs, startTime, endTime, file_ext='*.gz'):
    data=[]
    for sub_dir in sub_dirs:
#         print(sub_dir) no
        for fn in glob.glob(os.path.join(parent_dir,sub_dir, file_ext)):
            #print(fn)
            file=gzip.open(fn)
            reader = csv.reader(io.TextIOWrapper(file, newline=""))
            for row in reader:
                timestamp=float(row[0])
                if timestamp >=startTime and timestamp <=endTime:
                    x=row[2]
                    y=row[3]
                    z=row[4]
                    data.append([sub_dir,fn,timestamp,x,y,z])
                #print(timestamp,x,y,z)
            #break
        #break
    return data
        
import cv2

def parse_Video_files(parent_dir, sub_dirs, file_ext='*.mp4'):
    print(sub_dirs)
    
    
    video_IMU_sound_data=[]
    for sub_dir in sub_dirs:
#         for fn in tqdm(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)) ):
        for fn in glob.glob(os.path.join(parent_dir,sub_dir, file_ext)):
            print(fn)
            
            fn_list=fn.split('.')
            fn_sound=fn_list[0]+'.wav'
            if not os.path.exists(fn_sound):
                    os.system("ffmpeg -i {0} -ac 1 {1}".format(fn,fn_sound))
            
            Sound_raw_X, Sound_sample_rate = librosa.load(fn_sound)
            
            #Video End Time
            Endtime=os.path.getmtime(fn)*1000
            
            proc="ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {0}".format(fn)
            Duration= os.popen(proc).read()
            
            #Video Start Time
            startTime=Endtime - float(Duration)*1000
            
            
            #print(startTime,Endtime)
            # In this Start Time and End Time, find the IMU Samples and Store them
            data=parse_IMU_files(IMU_parent_dir,imu_sensor_dirs,startTime,Endtime)
            
            #Add the number of frame per video to the metadata
            cap = cv2.VideoCapture(fn)
            video_frames=0
            video_continue=True
            while video_continue:
                video_continue, img = cap.read()
                video_frames=video_frames+1
            
            #print(len(data))
#             print(video_frames)   no
            video_IMU_sound_data.append([fn,sub_dir,startTime,Endtime,data,Sound_sample_rate,Sound_raw_X,video_frames])
            
#             break
#         break
    return video_IMU_sound_data

video_IMU_sound_data=parse_Video_files(video_parent_dir,sub_video_dirs)
print(len(video_IMU_sound_data))

import pickle
with open('all_data_train.pkl', 'wb') as f:
    pickle.dump(video_IMU_sound_data, f)


video_IMU_sound_data=np.load('all_data_train.pkl')
# 3 = Acc Right Wrist
# 4 = GYRO Right Wrist
# 11 = Acc left Wrist
# 12 = GYRO left Wrist
# 35 = Activity Type phone
# 36 = Acc Phone
# 37 = Gyro Phone
# 38 = Compus Phone
# 42 = Step-Count Phone

# Trying: 3,4,11,12,36,37

#raw3 192
#raw4 192
#raw11 192
#raw12 192
#raw36 944
#raw37 938


#raw38 813
    
# For every video
def extract_IMU_sound_video(video_IMU_data):
    features = np.empty((0,Data_window_watch,12))
    labels=[]
    features_sound=np.empty((0,193))
    features_video =np.empty((0, Data_window_video, 64, 64, 3) )
    
    for i in range(len(video_IMU_data)):
        print(i)
        
        #Saving features after every 100 samples
        if (i+1)%100==0:
            features_video = features_video.astype('uint8')
            data=[features,labels,features_sound,features_video]
            with open('Data_train_'+str(i)+'ser.pkl', 'wb') as f:
                pickle.dump(data, f)
            features = np.empty((0,Data_window_watch,12))
            labels=[]
            features_sound=np.empty((0,193))
            features_video =np.empty((0, Data_window_video, 64, 64, 3) )
    
        
        #print(video_IMU_data[i][0])
        V_name=video_IMU_data[i][0]
        V_type=video_IMU_data[i][1]
        V_stime=video_IMU_data[i][2]
        V_etime=video_IMU_data[i][3]
        #print(V_name,V_etime-V_stime)
        duration=V_etime-V_stime
        
        
        #Sound Features
        Sound_X = video_IMU_data[i][6]
        Sound_sample_rate_expected= video_IMU_data[i][5]
        #print('Leng of sound:',len(Sound_X))
        sound_len=len(Sound_X)
        sound_sampling=(sound_len*1000)/duration #Duration is in milli seconds
        
        #Sound windows below
        sound_win=[]
        start=0
        end=start + sound_sampling * TimeWindow
        end=int(end)
        while end <= sound_len:
            winx=Sound_X[start:end]
            stft = np.abs(librosa.stft(winx))
            mfccs = np.mean(librosa.feature.mfcc(y=winx, sr=Sound_sample_rate_expected, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=Sound_sample_rate_expected).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(winx, sr=Sound_sample_rate_expected).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=Sound_sample_rate_expected).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(winx), sr=Sound_sample_rate_expected).T,axis=0)
            #print (start,end,len(winx))
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            sound_win.append(ext_features)
            start=start + sound_sampling * TimeWindow_slide
            start=int(start)
            end=start + sound_sampling * TimeWindow
            end=int(end)
        
        #Video Features and Windows
        video_len= video_IMU_data[i][7]
        #print(video_len,duration)
        video_sampling=(video_len*1000)/duration
        
        video_raw=[]
        start=0
        end=start + video_sampling * TimeWindow
        end=int(end)
        #print('Video Sampling Rate:',video_sampling)
        #print('Start is:',start,': End is:',end)
#         cap = cv2.VideoCapture(V_name)
#         video_continue=True
#         while video_continue:
#             video_continue, img = cap.read()
#             #video_raw.append(cv2.resize(img, (img_size, img_size)))
#             video_raw.append(cv2.resize(img, (img_size, img_size)))
# #             video_raw.append(img)
        cap = cv2.VideoCapture(V_name)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            video_raw.append(cv2.resize(img, (img_size, img_size)).astype('uint8'))
            
        
        Windows_video=[]  
        while end <= video_len:
            #print('Start is:',start,': End is:',end)
            win_video=video_raw[start:end]
            win_video=win_video[:Data_window_video]
            Windows_video.append(win_video)
            start=start + video_sampling * TimeWindow_slide
            start=int(start)
            end=start + video_sampling * TimeWindow
            end=int(end)
        
        #Dict of sensor data
        V_dict=dict()
        
        
        
        # Loop over the data items
        for j in range(len(video_IMU_data[i][4])):
            #print(video_IMU_data[i][4][j])
            S_type=video_IMU_data[i][4][j][0]
            timestamp=video_IMU_data[i][4][j][2]
            x=video_IMU_data[i][4][j][3]
            y=video_IMU_data[i][4][j][4]
            z=video_IMU_data[i][4][j][5]

            if S_type in V_dict:
                V_dict[S_type].append([timestamp,x,y,z])
            else:
                V_dict[S_type]=[]
                V_dict[S_type].append([timestamp,x,y,z])
            #print(S_type,timestamp,x,y,z)
    # 3 = Acc Right Wrist
    # 4 = GYRO Right Wrist
    # 11 = Acc left Wrist
    # 12 = GYRO left Wrist

        raw3=[]
        raw4=[]
        raw11=[]
        raw12=[]
        for sid in V_dict:
            if sid=='raw3':
                #print(sid,len(V_dict[sid]))
                # Timewindow 2 Sec, Sampling 25HZ.
                total_samples=len(V_dict[sid])
                sampling=(total_samples*1000)/(duration)
                #print('Sampling',sampling)
                start=0
                end=start + sampling * TimeWindow
                end=int(end)
                while end<=total_samples:
                    datawind=V_dict[sid][start:end]
                     #Fixing Datawind_leng Here:
                    datawind=datawind[:Data_window_watch]
                    raw3.append(datawind)

                    #print('window Samples:',len(datawind))
                    #winlen1.append(len(datawind))


                    #print(start,end)
                    start=start+(sampling)*(TimeWindow_slide)
                    start=int(start)
                    end=start + sampling * TimeWindow
                    end=int(end)

            if sid=='raw4':
                #print(sid,len(V_dict[sid]))
                # Timewindow 2 Sec, Sampling 25HZ.
                total_samples=len(V_dict[sid])
                sampling=(total_samples*1000)/(duration)
                #print('Sampling',sampling)
                start=0
                end=start + sampling * TimeWindow
                end=int(end)
                while end<=total_samples:
                    datawind=V_dict[sid][start:end]
                    #print('window Samples:',len(datawind))
                    #winlen2.append(len(datawind))
                    datawind=datawind[:Data_window_watch]
                    raw4.append(datawind)
                    #print(start,end)
                    start=start+(sampling)*(TimeWindow_slide)
                    start=int(start)
                    end=start + sampling * TimeWindow
                    end=int(end)

            if sid=='raw11':
                #print(sid,len(V_dict[sid]))
                # Timewindow 2 Sec, Sampling 25HZ.
                total_samples=len(V_dict[sid])
                sampling=(total_samples*1000)/(duration)
                #print('Sampling',sampling)
                start=0
                end=start + sampling * TimeWindow
                end=int(end)
                while end<=total_samples:
                    datawind=V_dict[sid][start:end]
                    #print('window Samples:',len(datawind))
                    #winlen3.append(len(datawind))
                    datawind=datawind[:Data_window_watch]
                    raw11.append(datawind)
                    #print(start,end)
                    start=start+(sampling)*(TimeWindow_slide)
                    start=int(start)
                    end=start + sampling * TimeWindow
                    end=int(end)

            if sid=='raw12':
                #print(sid,len(V_dict[sid]))
                # Timewindow 2 Sec, Sampling 25HZ.
                total_samples=len(V_dict[sid])
                sampling=(total_samples*1000)/(duration)
                #print('Sampling',sampling)
                start=0
                end=start + sampling * TimeWindow
                end=int(end)
                while end<=total_samples:
                    datawind=V_dict[sid][start:end]
                    datawind=datawind[:Data_window_watch]
                    raw12.append(datawind)

                    #print('window Samples:',len(datawind))
                    #winlen4.append(len(datawind))

                    #print(start,end)
                    start=start+(sampling)*(TimeWindow_slide)
                    start=int(start)
                    end=start + sampling * TimeWindow
                    end=int(end)
            #print(sid,len(V_dict[sid]))
        #print(len(raw3),len(raw4),len(raw11),len(raw12))
        raw3=np.array(raw3)
        raw4=np.array(raw4)
        raw11=np.array(raw11)
        raw12=np.array(raw12)
        sound_win=np.array(sound_win)
        Windows_video=np.array(Windows_video)
        
        # We abandon the timestamp
        #print(raw3.shape,raw4.shape,raw11.shape,raw12.shape)
        raw3_windows=raw3.shape[0]
        raw4_windows=raw4.shape[0]
        raw11_windows=raw11.shape[0]
        raw12_windows=raw12.shape[0]
        #print('sensor wind:',raw12_windows)
        sound_win_windows=sound_win.shape[0]
        video_win_windows=Windows_video.shape[0]
        #print('Sound wind:',sound_win_windows)
        #print('Video wind:',video_win_windows)
        try:
            #Sometimes Watch have less features, and we don't know the Reason
            min_features_watch=np.array([raw3.shape[1],raw4.shape[1],
                              raw11.shape[1],raw12.shape[1]]).min()
        
            if min_features_watch<Data_window_watch:
                print('Skipping:',raw3.shape,raw4.shape,raw11.shape,raw12.shape)
                continue
            min_features_Video=Windows_video.shape[1]
            if min_features_Video<Data_window_video:
                print('Skipping:',Windows_video.shape)
                continue
        except:
            print('Error Skipping:',raw3.shape,raw4.shape,raw11.shape,raw12.shape)
            continue
        #print(sound_win_windows)
        min_windows=np.array([raw3_windows,raw4_windows,
                              raw11_windows,raw12_windows,
                              sound_win_windows, 
                              video_win_windows]).min()
        
        if min_windows>0:
            output=np.concatenate((raw3[:min_windows,:,1:4], raw4[:min_windows,:,1:4],raw11[:min_windows,:,1:4],raw12[:min_windows,:,1:4]), axis=2)
            #print(output.shape)
            features=np.vstack([features,output])
           # print(sound_win[:min_windows].shape)
            features_sound=np.vstack([features_sound,sound_win[:min_windows]])
            #print(Windows_video[:min_windows].shape, features_video.shape)
            features_video=np.vstack([features_video,Windows_video[:min_windows]])
  
            
            #output.shape[0],V_type
            for k in range(output.shape[0]):
                labels.append(V_type)
        
    #     print(raw4.shape)
#         print(output.shape)
#         print(features.shape, features_sound.shape, Windows_video[:min_windows].shape )
#         break
        # 12-num of windows, 47-num of samples in window 12= 3*4 num of sensor reading types
    
    features_video = features_video.astype('uint8')
    data=[features,labels,features_sound,features_video]
    with open('Data_train_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Saved: '+ 'Data_train_'+str(i)+'.pkl')
                
    print(features_video.shape)
    return features,labels,features_sound, features_video


features,labels,features_sound, features_video=extract_IMU_sound_video(video_IMU_sound_data)

print('IMU features: ',features.shape,
      '\nLabels: ',len(labels),
      '\nSound features: ',features_sound.shape,
      '\nVideo features: ',features_video.shape)

# data=[features,labels,features_sound,features_video]

np.savez('extract_data/IMU_Sound_video_all_train.npz', features,labels,features_sound, features_video)