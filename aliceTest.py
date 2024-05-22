from EQTransformer.core.trainer import trainer
import os

trainer(input_hdf5=['/home/aliceyu/datasets/STEAD/chunk2/chunk2.hdf5', '/home/aliceyu/datasets/STEAD/chunk1/chunk1.hdf5'],
        input_csv=['/home/aliceyu/datasets/STEAD/chunk2/chunk2.csv', '/home/aliceyu/datasets/STEAD/chunk1/chunk1.csv'],
        output_name='test_trainer_chunk2and1',                
        cnn_blocks=2,
        lstm_blocks=1,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        label_type='gaussian',
        add_event_r=0.6,
        add_gap_r=0.2,
        shift_event_r=0.9,
        add_noise_r=0.5, 
        mode='generator',
        train_valid_test_split=[0.60, 0.20, 0.20],
        batch_size=20,
        epochs=10, 
        patience=2,
        gpuid=0,
        gpu_limit=None)



def test_generator():
    trainer(input_hdf5='./ModelsAndSampleData/100samples.hdf5',
        input_csv='./ModelsAndSampleData/100samples.csv',
        output_name='test_trainer',                
        cnn_blocks=2,
        lstm_blocks=1,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        label_type='gaussian',
        add_event_r=0.6,
        add_gap_r=0.2,
        shift_event_r=0.9,
        add_noise_r=0.5, 
        mode='generator',
        train_valid_test_split=[0.60, 0.20, 0.20],
        batch_size=20,                                                                                             
        epochs=10, 
        patience=2,
        gpuid=None,
        gpu_limit=None)

    dir_list = [ev for ev in os.listdir('.') if ev.split('_')[-1] == 'outputs']  
    if 'test_trainer_outputs'  in dir_list:
        successful = True
    else:
        successful = False        
        assert successful == True

#test_generator()
