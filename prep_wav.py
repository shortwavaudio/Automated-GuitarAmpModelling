from scipy.io import wavfile
import argparse
import numpy as np

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def main(args):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    X_all = in_data.astype(np.float32).flatten()  
    X_all = normalize(X_all).reshape(len(X_all),1)   
    y_all = out_data.astype(np.float32).flatten() 
    y_all = normalize(y_all).reshape(len(y_all),1)   


    # Get the last 20% of the wav data to run prediction and plot results
    y_train, y_val = np.split(y_all, [int(len(y_all)*.8)])
    x_train, x_val= np.split(X_all, [int(len(X_all)*.8)])


    save_wav(args.path + "/train/" + args.name + "-input.wav", x_train)
    save_wav(args.path + "/train/" + args.name + "-target.wav", y_train)

    #NOTE: The proper way to conduct training would be to have a different
    #      dataset for testing, so that the model tests on audio it hasn't
    #      seen during training. I resused the validation data here due to
    #      limited available audio data. 
    save_wav(args.path + "/test/" + args.name + "-input.wav", x_val)
    save_wav(args.path + "/test/" + args.name + "-target.wav", y_val)

    save_wav(args.path + "/val/" + args.name + "-input.wav", x_val) 
    save_wav(args.path + "/val/" + args.name + "-target.wav", y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("name")
    parser.add_argument("--path", type=str, default="Data")


    args = parser.parse_args()
    main(args)