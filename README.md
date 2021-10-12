# dreamer-torch

Pytorch version of Dreamer, which follows the original TF v2 codes (https://github.com/danijar/dreamerv2/tree/faf9e4c606e735f32c8b971a3092877265e49cc2).

Due to limitation of resource, we tested it with the official TF version2 codes for 10 tasks (5 DMC / 5 Atari tasks) as below.

![alt text](https://github.com/jsikyoon/dreamer-torch/blob/main/figs/dreamer-test.png?raw=true)

As we can see, for almost tasks, it shows similar performance with Dreamer when comparing the reported performance on paper and from running codes.

For freeway, I found that it is slower than Dreamer, maybe I think it is because of random seed. I will find the reason more.

The below logs are from Tensorboard, you also can see through [Tensorboard logs](https://tensorboard.dev/experiment/QsJYF8DaTaaLiPJFvqp02Q/#scalars).

- DMC walker walk task

![alt text](https://github.com/jsikyoon/dreamer-torch/blob/main/figs/walker_walk_logs.png?raw=true)


- Atari bank heist task

![alt text](https://github.com/jsikyoon/dreamer-torch/blob/main/figs/bank_heist_logs.png?raw=true)


- atari freeway task

![alt text](https://github.com/jsikyoon/dreamer-torch/blob/main/figs/freeway_logs.png?raw=true)

## How to use
For required packages to run this, you can find from [requirements.txt](https://github.com/jsikyoon/dreamer-torch/blob/main/requirements.txt). 

The command to run is exactly same to the official codes, [you can use it](https://github.com/danijar/dreamerv2/tree/faf9e4c606e735f32c8b971a3092877265e49cc2#instructions). 

## Contact
Any feedback are welcome! Please open an issue on this repository or send email to Jaesik Yoon (jaesik.yoon.kr@gmail.com).
