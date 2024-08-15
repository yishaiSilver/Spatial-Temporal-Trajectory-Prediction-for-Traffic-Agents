import torch


def collate_fn(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    print(batch)
    city = [scene['city'] for scene in batch]
    scene_idx = [scene['scene_idx'] for scene in batch]
    agent_id = [scene['agent_id'] for scene in batch]
    car_mask = [scene['car_mask'] for scene in batch]
    track_id = [scene['track_id'] for scene in batch]
    pin = [scene['p_in'] for scene in batch]
    vin = [scene['v_in'] for scene in batch]
    pout = [scene['p_out'] for scene in batch]
    vout = [scene['v_out'] for scene in batch]
    lane = [scene['lane'] for scene in batch]
    lane_norm = [scene['lane_norm'] for scene in batch]
    
    
    return [(city, scene_idx, agent_id, car_mask, track_id, pin, vin, pout, vout, lane), lane_norm]