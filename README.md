# SoundActions

Code repository for paper "Investigating Aesthetic Audio-Visual Perception of Multi-Model Neural Networks with the SoundActions Dataset".

## Label cleaning
- Shaeffer type: majority vote, unknown if no majority
- Material: count >=1, 15 classes
    - metal, wood, plastic, organic, skin, stone, fabric, rubber, liquid, snow, ceramic, paper, leather,electronic, others
- Environment: majority vote, fallback to indoor others if no majority
    - outdoor, office, room, corridor, hall, bathroom, kitchen, car, indoor others
- Enjoyability: majority vote, unknown if no majority


## Requirements
- torch
- torchvision
- torchaudio