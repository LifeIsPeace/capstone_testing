# Notes
Keep the model very simple at first just to get a feel for what can go wrong. There's so much trouble shooting to do

Right now the plan is to take a mel spectrogram and predict which pitches (88 piano keys) are active at each time frame

```
mel spectrogtam (frequency x time)
then
CNN extracts pitch-relevant patterns along frequency
then
For each time frame -> predict 88 pitch activations
```

The model needs to ensure time resolution is unaffected

So the goal is to compress frequency but preserve time and classify pitch per frame

## Remember this is a multi-label classification (not multi-class meaning exactly one class is correct)
The sigmoid function gives the probability of a note being on or off. Remember the network outputs logits (unnomalized final scores of the model)
The sigmoid turns it into a probability that a pitch is active

BCEWithLogitsLoss is commonly used. Research later

The Adam (Adpative Moment Estimation) optimizer is standard for audio. We need to choose a good learning rate. Play around with it