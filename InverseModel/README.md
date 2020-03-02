# Data description

image_0-7: arm no reset

image_10-17: arm reset but occasionally unreachable

image_20-27, 30-37: arm reset always reachable

image_28-29: long range data; 28 all, 29 first 1k data no collision

image_38-39: even longer range data; 38 all, 39 first 1k data no collision

image_40_48: larger box

Learning (normalize data to (-1, 1), mean: [-0.908,  0.356, -0.908], std: [0.144, 0.069, 0.158]):

image_20: 0.07556088717675045
image_21: 0.03013499572977911 (training)
image_28: 0.0963768486964377
image_38: 0.13977662882761696

Base:

image_20: 0.13977662882761696
image_21: 0.09778188921837593
image_28: 0.12106623075863897
image_38: 0.12375992358185979

Learning with 100K

image_20: 0.05323721414637045; 0.02910992555545167
image_21: 0.01757739567784716 (training); 0.01640840114215274

Learning (normalize data to (-5, 5), mean: [ 0.001, -0.001, -0.001], std: [0.998, 1.022, 1.012]):

image_20: 0.03995610861120316
image_21: 0.02433286132548355 (training)

image_50-59: data with new end-effector, 52, 55, 58 have some defects. 23k each.


Learning with 100K; amortizing learning rate; comparing world coordinates predictions with pixel coordinates prediction

image_49: (new depth image) 100 (test trials)

world_116: 0.04543
world_117: 0.04619
world_118: 0.06644
world_125: 0.04556
world_126: 0.04447

gt_world_116: 0.04485
gt_world_117: 0.04453
gt_world_118: 0.04489
gt_world_125: 0.04541
gt_world_126: 0.04458

pixel_122: 0.04555
pixel_123: 0.04611
pixel_124: 0.04525
pixel_129: 0.04438
pixel_130: 0.04509

gt_pixel_122: 0.04609
gt_pixel_123: 0.04715
gt_pixel_124: 0.04656
gt_pixel_129: 0.04634
gt_pixel_130: 0.04547


image_70: 17k
image_71: 2k
image_72: 55k
image_73: 15k
image_74: 17k
image_75: 8k
image_76: 55k
image_77: 55k
image_78: 55k
image_79: 46k


poke_data.gif: replay from bad data
replaypoke.gif: replay from good data
stochastic_poke.gif: red block and blue block overlapping 


good_data: no unreachable errors, max_arm_reach set to 0.90
image_80: 1k
image_81: 44k
image_82: 30k 
image_83: 13k
image_84: 21k


image_85-88: 70k, no unreachable errors, max_arm_reach set to 0.91, x_max boundary set to 0.75
