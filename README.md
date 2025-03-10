# brainwide-RRR-encoding-model

🚧 *This project is under active development. Stay tuned for updates!* 🚧

## Overview

This project implements a simple but efficient linear encoding model for describing the single-neuron, task-driven responses.
See (Methods of) [1] for detailed information about the implemented encoding model.

## Current Status

The core functions are implemented in `RRRGD.py` and `RRRGD_main_CV.py`. 

If you are interested in fitting the model to **your data**, you can check the `example0` folder for a minimum, toy example of fitting a reduced-rank regression (RRR) encoding model to an example IBL session with 100 trials. I attached the data in the `example0` folder so that you can run the code directly and get a hands-on experience.

If you are interested in fitting the model to [**IBL data**](https://viz.internationalbrainlab.org/), you can check the `example1` folder for a comprehansive example of fitting a reduced-rank regression (RRR) encoding model to a large dataset including 100+ session Neuropixels recordings of 10,000+ neurons. 

In case you have any questions, please get in touch with me: shuqi.wang@epfl.ch

## How to Stay Updated

-  ⁠🌟 *Star this repository* to receive notifications about the latest updates.
-  ⁠📧 *Watch* the repository for detailed development progress.

---

Thank you for your interest! Stay tuned for more updates. 


### References:
[1] Posani, L., Wang, S., Muscinelli, S. P., Paninski, L., & Fusi, S. (2025). Rarely categorical, always high-dimensional: how the neural code changes along the cortical hierarchy. bioRxiv, 2024-11.