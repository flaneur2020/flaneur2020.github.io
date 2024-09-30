https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models

- The Diffusion process consists of **forward diffusion** and **reverse diffusion**
- Forward diffusion is used to add noise to the input image using a **schedule**
- There are different types of schedules (we’ve used linear), and they decide how much noise is added at the given step **t**
- We don’t have to use an iterative process to add noise, it can be done in one step with the equation described in the **[Real noising process](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models#real-noising-process-only-last-equation-is-important)** section
- Reverse diffusion consists of multiple steps in which a **small amount of noise is removed at every step** (**[equation](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models#some-math-you-can-skip-but-probably-worth-reading)**)
- <mark>**Diffusion model predicts the entire noise**</mark>, not the difference between step **t** and **t-1**
- You might use a different schedule for inference than for training (including a different number of steps)
- Diffusion model uses a **modified U-Net architecture**
- To add information about timestep **t** and prompt, we use **[sinusoidal encoding](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers#positional-encoding-visualization)** and **[text embedder](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models#embedder)**
- Some of the ResNet blocks are replaced with **Self-Attention blocks**