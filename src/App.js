import React from 'react';
import Navbar from './components/Navbar';
import ParentSection from './components/ParentSection';
import 'katex/dist/katex.min.css';

function App() {
  const parentSections = [
    {
      id: 'main-functions',
      title: 'Main Functions',
      sections: [
        {
          id: 'some-samples',
          title: 'Some Samples',
          content: [
            {
              type: 'paragraph',
              text: 'We are using the DeepFloyd IF model. We use 2 stages of the model:'
            },
            {
              type: 'list',
              items: [
                'Stage I: The base model generates initial low-resolution images 64x64 from a given text prompt.',
                'Stage II: The super-resolution (SR) model takes the low-resolution images from Stage I and enhances them to 256x256 pixels. This stage ensures high-quality details and sharpness in the generated images.'
              ]
            },
            {
              type: 'paragraph',
              text: 'Here are some sample images generated using the DeepFloyd IF model with the respective text prompts. We used a seed of 222.'
            },
            {
              type: 'image-grid',
              columns: 3,
              images: [
                {
                  title: 'an oil painting of a snowy mountain village (Stage I, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_0_5_stepsan_oil_painting_of_a_snowy_mountain_village.png`
                },
                {
                  title: 'a man wearing a hat (Stage I, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_1_5_stepsa_man_wearing_a_hat.png`
                },
                {
                  title: 'a rocket ship (Stage I, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_2_5_stepsa_rocket_ship.png`
                },
                {
                  title: 'an oil painting of a snowy mountain village (Stage II, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_0_5_stepsan_oil_painting_of_a_snowy_mountain_village.png`
                },
                {
                  title: 'a man wearing a hat (Stage II, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_1_5_stepsa_man_wearing_a_hat.png`
                },
                {
                  title: 'a rocket ship (Stage II, 5 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_2_5_stepsa_rocket_ship.png`
                },
                {
                  title: 'an oil painting of a snowy mountain village (Stage I, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_0_100_stepsan_oil_painting_of_a_snowy_mountain_village.png`
                },
                {
                  title: 'a man wearing a hat (Stage I, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_1_100_stepsa_man_wearing_a_hat.png`
                },
                {
                  title: 'a rocket ship (Stage I, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage1_2_100_stepsa_rocket_ship.png`
                },
                {
                  title: 'an oil painting of a snowy mountain village (Stage II, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_0_100_stepsan_oil_painting_of_a_snowy_mountain_village.png`
                },
                {
                  title: 'a man wearing a hat (Stage II, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_1_100_stepsa_man_wearing_a_hat.png`
                },
                {
                  title: 'a rocket ship (Stage II, 100 inference steps)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/stage2_2_100_stepsa_rocket_ship.png`
                },
              ]
            },
            {
              type: 'paragraph',
              text: 'We observe that in each of the cases, the image generated matches the text prompt. Stage I and Stage II are also coherent with each other, with Stage II being a higher resolution version of Stage I. With more inference steps, we observe better quality image (less noise, more details), especially on the Stage II output.'
            },
          ]
        },
        {
          id: 'forward-process',
          title: 'Forward Process',
          content: [
            {
              type: 'paragraph',
              text: 'The forward process progressively adds noise to a clean image, transforming it into pure noise over multiple time steps. This process is defined as a conditional distribution where the noisy image at time step t, denoted as xt, is generated from the clean image x0.'
            },
            {
              type: 'math',
              text: '\\( q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) I) \\)'
            },
            {
              type: 'paragraph',
              text: 'This can also be expressed as a sampling equation, combining a scaled version of the clean image and Gaussian noise.'
            },
            {
              type: 'math',
              text: '\\( x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon \\)'
            },
            {
              type: 'math',
              text: '\\( \\epsilon \\sim \\mathcal{N}(0, I) \\)'
            },
            {
              type: 'paragraph',
              text: 'Here is a test image at different noise levels (t=250, t=500 and t=750)',
            },
            {
              type: 'image-grid',
              columns: 4,
              images: [
                {
                  title: 'Campanile',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_resized.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 250)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_250.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 500)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_500.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 750)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_750.png`
                },
              ]
            },
          ]
        },
        {
          id: 'classical-denoising',
          title: 'Classical Denoising',
          content: [
            {
              type: 'paragraph',
              text: 'One way to deonise an image is by applying a Gaussian blur. This is done by convolving the noisy image with a Gaussian kernel. We used a 5x5 Gaussian Kernel of sigma=1. Here are the results:'
            },
            {
              type: 'image-grid',
              columns: 3,
              images: [
                {
                  title: 'Noised Campanile, (Timestep 250)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_250.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 500)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_500.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 750)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_750.png`
                },
                {
                  title: 'Gaussian Denoised Campanile, (Timestep 250)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/blur_denoised_image_t250.png`
                },
                {
                  title: 'Gaussian Denoised Campanile, (Timestep 500)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/blur_denoised_image_t500.png`
                },
                {
                  title: 'Gaussian Denoised Campanile, (Timestep 750)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/blur_denoised_image_t750.png`
                },
              ]
            }
          ]
        },
        {
          id: 'one-step-denoising',
          title: 'One-Step Denoising',
          content: [
            {
              type: 'paragraph',
              text: 'In one-step denoising, we use the base model to predict and remove noise from a noisy image at a given timestep. The model estimates the noise present in the noisy image and subtracts it to reconstruct the denoised image:'
            },
            {
              type: 'math',
              text: '\\( \\hat{x}_0 = \\frac{x_t - \\sqrt{1 - \\bar{\\alpha}_t} \\cdot \\hat{\\epsilon}}{\\sqrt{\\bar{\\alpha}_t}} \\)'
            },
            {
              type: 'paragraph',
              text: 'Since the model expects a text condition, we used "a high quality photo". Here are some results:'
            },
            {
              type: 'image-grid',
              columns: 4,
              images: [
                {
                  title: 'Original Campanile',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_resized.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 250)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_250.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 500)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_500.png`
                },
                {
                  title: 'Noised Campanile, (Timestep 750)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_timestep_750.png`
                },
                {
                  title: 'Original Campanile',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_resized.png`
                },
                {
                  title: 'One-Step Denoised Campanile, (Timestep 250)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/denoised_image_t250.png`
                },
                {
                  title: 'One-Step Denoised Campanile, (Timestep 500)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/denoised_image_t500.png`
                },
                {
                  title: 'One-Step Denoised Campanile, (Timestep 750)',
                  imageUrl: `${process.env.PUBLIC_URL}/images/denoised_image_t750.png`
                },
              ]
            }
          ]
        },
        {
          id: 'iterative-denoising',
          title: 'Iterative Denoising',
          content: [
            {
              type: 'paragraph',
              text: 'In iterative denoising, we gradually refine the noisy image by applying multiple reverse steps through the reverse diffusion process. Starting at timestep t = 690, we denoise the image in strides of size 30, iterating down to t = 0. The update equation is as follows:'
            },
            {
              type: 'math',
              text: '\\( x_{t\'} = \\frac{\\sqrt{\\bar{\\alpha}_{t\'} } \\beta_t}{1 - \\bar{\\alpha}_t} \\hat{x}_0 + \\frac{\\sqrt{\\alpha_t} (1 - \\bar{\\alpha}_{t\'})}{1 - \\bar{\\alpha}_t} x_t + v_\\sigma \\)'
            },
            {
              type: 'paragraph',
              text: 'Since the model expects a text condition, we used "a high quality photo". Here are the results of the iterative denoising process, along with the single-step denoise and gaussian blur denoise for comparison:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                {
                  title: 'Comparison of Denoising',
                  imageUrl: `${process.env.PUBLIC_URL}/images/reconcatenated_image.jpg`
                },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                {
                  title: 'Original Campanile',
                  imageUrl: `${process.env.PUBLIC_URL}/images/campanile_resized.png`
                },
              ]
            }
          ]
        },
        {
          id: 'diffusion-sampling',
          title: 'Diffusion Model Sampling',
          content: [
            {
              type: 'paragraph',
              text: 'We now aim to generate images from scratch, starting with pure noise instead of a noised image. Beginning with random noise at t = 990, we denoise the image in strides of 30 in the reverse diffusion process until we reach t = 0. Similarly, we used the text prompt "a high quality photo". Here are some results:'
            },
            {
              type: 'image-grid',
              columns: 5,
              images: [
                { title: 'Example 1', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_0.png` },
                { title: 'Example 2', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_1.png` },
                { title: 'Example 3', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_2.png` },
                { title: 'Example 4', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_3.png` },
                { title: 'Example 5', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_4.png` },
              ]
            },
          ]
        },
        {
          id: 'classifier-guidance',
          title: 'Classifier Free Guidance',
          content: [
            {
              type: 'paragraph',
              text: 'To improve the quality of the generated images, we use a technique called Classifier-Free Guidance (CFG). In this method, at each denoising step, we run the diffusion model twice: once with the text prompt "a high quality photo" (the conditional prompt) and once with a null prompt "" (the unconditional prompt). The model predicts the noise for each run, and we combine these predictions using the following equation:'
            },
            {
              type: 'math',
              text: '\\( \\epsilon = \\epsilon_u + \\gamma (\\epsilon_c - \\epsilon_u) \\)'
            },
            {
              type: 'paragraph',
              text: 'Here, we used a gamma value of 7. Here are some results of images generated:'
            },
            {
              type: 'image-grid',
              columns: 5,
              images: [
                { title: 'Example 1 (cfg)', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_cfg_0.png` },
                { title: 'Example 2 (cfg)', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_cfg_1.png` },
                { title: 'Example 3 (cfg)', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_cfg_2.png` },
                { title: 'Example 4 (cfg)', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_cfg_3.png` },
                { title: 'Example 5 (cfg)', imageUrl: `${process.env.PUBLIC_URL}/images/generated_from_scratch/generated_image_cfg_4.png` },
              ]
            }
          ]
        },
      ],
    },
    {
      id: 'more-applications',
      title: 'More Applications',
      sections: [
        {
          id: 'image-to-image',
          title: 'Image to Image Translation',
          content: [
            {
              type: 'paragraph',
              text: 'In this section, we explore image-to-image translation using Classifier-Free Guidance (CFG). The goal is to make controlled edits to an existing image by adding noise and then progressively denoising. By introducing varying levels of noise, we can create edits ranging from subtle changes to significant transformations. This process utilizes the denoising capabilities of the diffusion model, which effectively "forces" the noisy image back onto the natural image manifold. Once again, we used the prompt "a high quality photo"'
            },
            {
              type: 'paragraph',
              text: 'Here are the results of applying this approach at different noise levels, labeled by the number of denoising steps performed, starting from indices [1, 3, 5, 7, 10, 20], which correspond to timesteps [960, 900, 840, 780, 690, 390] respectively:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Starting index 1 (left) to 20 (second from right) and original (rightmost)', imageUrl: `${process.env.PUBLIC_URL}/images/reconstruct.jpg` },
              ]
            },
            {
              type: 'paragraph',
              text: 'We can also use this procedue to generate images from nonrealistic images (such as drawings) to the natural image manifold: '
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Starting index 1 (left) to 20 (second from right) and original (rightmost)', imageUrl: `${process.env.PUBLIC_URL}/images/drawing.jpg` },
              ]
            },
            {
              type: 'paragraph',
              text: 'We can also apply this procedure to the task of image inpainting. Inpainting involves reconstructing or generating missing or corrupted parts of an image while preserving the original content in other areas. Using the diffusion-based approach, we can replace parts of the image defined by a binary mask m, where m = 1 represents areas to be inpainted and m = 0 represents areas to retain. During the denoising process, at each timestep, we adjust the intermediate noisy image to ensure it matches the original image outside the inpainted region, as follows:'
            },
            {
              type: 'math',
              text: '\\( x_t \\leftarrow \\mathbf{m} x_t + (1 - \\mathbf{m}) \\cdot \\text{forward}(x_{\\text{orig}}, t) \\)',
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Inpainting', imageUrl: `${process.env.PUBLIC_URL}/images/inpaint.jpg` },
              ]
            },
            {
              type: 'paragraph',
              text: 'We can also apply this procedure with text condition. Here, we used the new prompt "a rocket ship":'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Starting index 1 (left) to 20 (second from right) and original (rightmost)', imageUrl: `${process.env.PUBLIC_URL}/images/rocket.jpg` },
              ]
            },
          ]
        },
        {
          "id": "visual-anagrams",
          "title": "Visual Anagrams",
          "content": [
            {
              "type": "paragraph",
              "text": "In this section, we explore creating visual anagrams using diffusion models. A visual anagram appears as one image in its original orientation but transforms into another image when flipped upside down. For example, it might look like 'an oil painting of an old man' in one orientation and 'an oil painting of people around a campfire' when flipped."
            },
            {
              "type": "paragraph",
              "text": "To achieve this, we use the denoising process with two text prompts: one for the original orientation and one for the flipped orientation. At each timestep, we denoise the image using the first prompt to estimate the noise, flip the image, and denoise it again using the second prompt. We then average the two noise estimates and proceed with the denoising process."
            },
            {
              "type": "math",
              "text": "\\( \\epsilon_1 = \\text{UNet}(x_t, t, p_1) \\)"
            },
            {
              "type": "math",
              "text": "\\( \\epsilon_2 = \\text{flip}(\\text{UNet}(\\text{flip}(x_t), t, p_2)) \\)"
            },
            {
              "type": "math",
              "text": "\\( \\epsilon = \\frac{\\epsilon_1 + \\epsilon_2}{2} \\)"
            },
            {
              "type": "paragraph",
              "text": "Below are some examples of visual anagrams generated using this method."
            },
            {
              type: 'image-grid',
              columns: 2,
              images: [
                { title: 'an oil painting of an old man', imageUrl: `${process.env.PUBLIC_URL}/images/effects/oldman.jpg` },
                { title: 'an oil painting of people around a campfire', imageUrl: `${process.env.PUBLIC_URL}/images/effects/campfire.jpg` },
                { title: 'a photo of a hipster barista', imageUrl: `${process.env.PUBLIC_URL}/images/effects/barista.jpg` },
                { title: 'a photo of a dog', imageUrl: `${process.env.PUBLIC_URL}/images/effects/dog.jpg` },
                { title: 'a photo of a dog', imageUrl: `${process.env.PUBLIC_URL}/images/effects/dog2.jpg` },
                { title: 'a rocket ship', imageUrl: `${process.env.PUBLIC_URL}/images/effects/rocket.jpg` },
              ]
            },
          ]
        },
        {
          "id": "hybrid-images",
          "title": "Hybrid Images",
          "content": [
            {
              "type": "paragraph",
              "text": "In this section, we explore the creation of hybrid images using diffusion models. A hybrid image combines low-frequency details from one image with high-frequency details from another, resulting in a composite that appears differently when viewed at varying distances. This technique allows us to merge content in a visually compelling way, leveraging the unique capabilities of diffusion models."
            },
            {
              "type": "paragraph",
              "text": "To generate hybrid images, we use the U-Net diffusion model to estimate noise for two different text prompts. We then separate the low-frequency and high-frequency components of the two noise estimates. By combining the low frequencies from one prompt and the high frequencies from the other, we create a composite noise estimate that guides the denoising process."
            },
            {
              "type": "math",
              "text": "\\( \\epsilon_1 = \\text{UNet}(x_t, t, p_1) \\)"
            },
            {
              "type": "math",
              "text": "\\( \\epsilon_2 = \\text{UNet}(x_t, t, p_2) \\)"
            },
            {
              "type": "math",
              "text": "\\( \\epsilon = f_{\\text{lowpass}}(\\epsilon_1) + f_{\\text{highpass}}(\\epsilon_2) \\)"
            },
            {
              "type": "paragraph",
              "text": "Here are some results. The first prompt is the one where we observe from far and the second prompt is the one we observe close up. Note: to observe the effects of the low frequency image more clearly, please view it from far away."
            },
            {
              type: 'image-grid',
              columns: 3,
              images: [
                { title: 'a lithograph of a skull; a lithograph of waterfalls', imageUrl: `${process.env.PUBLIC_URL}/images/effects/skull_waterfall.png` },
                { title: 'an oil painting of an old man; an oil painting of people around a campfire', imageUrl: `${process.env.PUBLIC_URL}/images/effects/oldman_campfire.jpg` },
                { title: 'a rocket ship; a photo of a dog', imageUrl: `${process.env.PUBLIC_URL}/images/effects/rocket_dog.jpg` },
              ]
            },
          ]
        }
      ],
    },
    {
      id: 'single-step-unet',
      title: 'Single Step Denoising UNet',
      sections: [
        {
          id: 'architecture',
          title: 'Architecture',
          content: [
            {
              type: 'paragraph',
              text: 'The Single Step Denoising UNet is designed to estimate and remove noise from images in a single reverse step. The model uses the following UNet architecture.'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'UNet Architecture', imageUrl: `${process.env.PUBLIC_URL}/images/unet.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'UNet Blocks', imageUrl: `${process.env.PUBLIC_URL}/images/blocks.png` },
              ]
            },
          ],
        },
        {
          id: 'training',
          title: 'Training',
          content: [
            {
              type: 'paragraph',
              text: 'To train the UNet as a single-step denoiser, we first add noise to the clean image. Specifically, we use a noise level of  = 0.5. The noisy image z is generated from the clean image x using the following formula:'
            },
            {
              type: 'math',
              text: '\\( z = x + \\sigma \\epsilon \\), where \\( \\epsilon \\sim \\mathcal{N}(0, I) \\)'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Noised Images for different sigma', imageUrl: `${process.env.PUBLIC_URL}/images/noising_process_visualization.jpg` },
              ]
            },
            {
              type: 'paragraph',
              text: 'The UNet takes the noisy image z as input and attempts to reconstruct the original clean image x in a single step. The Loss function is the L2 loss between the reconstructed and original images, as follows:'
            },
            {
              type: 'math',
              text: '\\( \\mathcal{L} = \\mathbb{E}_{z, x} \\| D_{\\theta}(z) - x \\|^2 \\)'
            },
            {
              type: 'paragraph',
              text: 'We train the model on the MNIST dataset using a batch size of 256 and a hidden dimension of D = 128. The optimizer is Adam with a learning rate of 10e-4, and the model is trained for 5 epochs. Here is the training loss curve:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Loss per step (log scale)', imageUrl: `${process.env.PUBLIC_URL}/images/training_loss_curve_step_log.png` },
              ]
            },
          ]
        },
        {
          id: 'sampling',
          title: 'Sampling',
          content: [
            {
              type: 'paragraph',
              text: 'Here are the the results on digits of the test set after Epoch 1 and Epoch 5. The images in the test set are noised with sigma = 0.5 before being passed into the model.'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 1', imageUrl: `${process.env.PUBLIC_URL}/images/epoch1.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 5', imageUrl: `${process.env.PUBLIC_URL}/images/epoch5.png` },
              ]
            },
            {
              type: 'paragraph',
              text: 'Our model is trained to denoise the digits which are noised with sigma = 0.5. Here, we test how well our model (after Epoch 5) can reconstruct images for different sigma values:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Out of distribution testing', imageUrl: `${process.env.PUBLIC_URL}/images/out_of_distribution_samples.jpg` },
              ]
            },
          ],
        },
        {
          id: 'starting_from_pure_noise',
          title: 'Starting From Pure Noise',
          content: [
            {
              type: 'paragraph',
              text: 'We now attempt to train the model, replacing z with pure noise instead of a noised version of x. In this setting, the model is trained to map pure noise to the clean image. However, since the input noise z is independent of the target image x, the model has no information to infer the specific details of x and, in an MSE sense, the optimal solution is to output a constant value – namely, the mean of the training images.'
            },
            {
              type: 'math',
              text: '\\( \\mathcal{L}(D) = \\mathbb{E}_{z,x} \\Bigl[ \\| D(z) - x \\|^2 \\Bigr] \\)'
            },
            {
              type: 'paragraph',
              text: 'Here, D(z) is the denoiser and z is drawn from a fixed standard gaussian distribution independent of x. Because z carries no information about x, the best the model can do is output the same constant c for all z. Thus, we set:'
            },
            {
              type: 'math',
              text: '\\( D(z) = c \\quad \\text{for all } z \\)'
            },
            {
              type: 'paragraph',
              text: 'Substituting this constant predictor into the loss gives:'
            },
            {
              type: 'math',
              text: '\\( \\mathcal{L}(c) = \\mathbb{E}_{x} \\Bigl[ \\| c - x \\|^2 \\Bigr] \\)'
            },
            {
              type: 'paragraph',
              text: 'To find the optimal constant c, we differentiate with respect to c:'
            },
            {
              type: 'math',
              text: '\\( \\frac{d}{dc} \\mathcal{L}(c) = 2\\Bigl(c - \\mathbb{E}[x]\\Bigr) = 0 \\quad \\Rightarrow \\quad c = \\mathbb{E}[x] \\)'
            },
            {
              type: 'paragraph',
              text: 'Thus, if the input noise is independent of x, the MSE-optimal denoiser is simply to output the mean of the clean images. This explains why, in one-step denoising from pure noise, the model learns to predict the mean of the test set.'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                {
                  title: 'Epoch 1',
                  imageUrl: `${process.env.PUBLIC_URL}/images/pure_noise_denoising_epoch_1.jpg`
                },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                {
                  title: 'Epoch 1',
                  imageUrl: `${process.env.PUBLIC_URL}/images/pure_noise_denoising_epoch_5.jpg`
                },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                {
                  title: 'Epoch 1',
                  imageUrl: `${process.env.PUBLIC_URL}/images/average_image.png`
                },
              ]
            },
          ]
        },
      ],
    },
    {
      id: 'time-conditioned-ddpm',
      title: 'Time-Conditioned DDPM Model',
      sections: [
        {
          id: 'time_cond_architecture',
          title: 'Architecture',
          content: [
            {
              type: 'paragraph',
              text: 'The Time-Conditioned DDPM model extends the UNet architecture by conditioning the model on a time step input. This input allows the model to account for the varying levels of noise present at different timesteps during the forward diffusion process. Our model will have a total of 300 timesteps. The timestep is injected into the model as follows:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Injecting time condition', imageUrl: `${process.env.PUBLIC_URL}/images/timeconditioned.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'FC Block', imageUrl: `${process.env.PUBLIC_URL}/images/fcblock.png` },
              ]
            },
            { type: 'paragraph', 
              text: 'To inject the timestep t into the model, we first normalize it and pass it through a fully connected block (FCBlock), which outputs a tensor of shape 2D by 1 by 1 or 1D by 1 by 1. This tensor is then added elementwise to the respective layer’s feature maps, which are broadcasted across the spatial dimensions to match the feature maps’s H and W.' 
            },
          ],
        },
        {
          id: 'time_cond_training',
          title: 'Training',
          content: [
            {
              type: 'paragraph',
              text: 'We sample x0 (clean image), timestep t, and epsilon. We add noise to the image according to the following to get xt:'
            },
            {
              type: 'math',
              text: '\\( x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon \\), where \\( \\epsilon \\sim \\mathcal{N}(0, I) \\)'
            },
            {
              type: 'paragraph',
              text: 'We pass xt and t into the model, and the model outputs a prediction for epsilon. We calculate the loss as:'
            },
            {
              type: 'math',
              text: '\\( \\mathcal{L} = \\mathbb{E}_{x_0, t, \\epsilon} \\left[ \\| \\epsilon - \\hat{\\epsilon}_\\theta(x_t, t) \\|^2 \\right] \\)'
            },
            {
              type: 'paragraph',
              text: 'We train the model on the MNIST dataset using a batch size of 128 and a hidden dimension of D = 64. The optimizer is Adam with initial learning rate of 10e-3 and an exponential learning rate decay scheduler. The model is trained for 20 epochs. Here is the training loss curve:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Loss per step (log scale)', imageUrl: `${process.env.PUBLIC_URL}/images/step_loss_curve.png` },
              ]
            },
          ],
        },        
        {
          id: 'time_cond_sampling',
          title: 'Sampling',
          content: [
            {
              type: 'paragraph',
              text: 'The sampling starts from pure noise and iteratively denoises from t = 299 to t = 0. At each timestep, the model predicts the noise and updates the image according to the following update equation:'
            },
            {
              type: 'math',
              text: '\\( \\mu_t = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{1 - \\alpha_t}{\\sqrt{1 - \\bar{\\alpha}_t}} \\hat{\\epsilon}_\\theta(x_t, t) \\right) \\)'
            },
            {
              type: 'math',
              text: '\\( x_{t-1} = \\mu_t + \\sqrt{\\beta_t} z, \\text{where } z \\sim \\mathcal{N}(0, I) \\text{ if } t > 1, \\text{ else } z = 0 \\)'
            },
            {
              type: 'paragraph',
              text: 'This iterative process continues until timestep t = 0, producing a denoised image as the final output. Here are some results for various Epochs.'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Time-conditioned DDPM Outputs', imageUrl: `${process.env.PUBLIC_URL}/images/stacked_samples_with_labels.png` },
              ]
            },
          ],
        },        
      ],
    },
    {
      id: 'class-and-time-conditioned-ddpm',
      title: 'Class and Time Conditioned DDPM Model',
      sections: [
        {
          id: 'time_class_architecture',
          title: 'Architecture',
          content: [
            {
              type: 'paragraph',
              text: 'To inject The Class Conditioning, we first one-hot encode the class into a vector, and then pass it into a FC Block (similar to how we got the time embedding). Instead of using the output of the FC Block to shift the values of the layers they are injected into, we instead use the output here to scale the layers. We inject them at the same layers where we injected the time conditioning.'
            },
          ],
        },
        {
          id: 'time_class_training',
          title: 'Training',
          content: [
            {
              type: 'paragraph',
              text: 'The training process incorporates both time-step conditioning and class conditioning, allowing the model to generate images aligned with specific classes while accounting for noise levels at different timesteps. The model is trained on a class-labeled dataset, where the input to the model consists of noised image, their corresponding class labels, and timestep t.'
            },
            {
              type: 'paragraph',
              text: 'We use Classifier-Free Guidance (CFG) during training. This is achieved by randomly dropping the class conditioning with a probability of 0.1. When the class conditioning is dropped, we replace the class embedding with a zero vector to represent a "null class." This ensures the model can perform both class-conditioned and unconditional generation effectively. Here is the training loss curve:'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Loss per step (log scale)', imageUrl: `${process.env.PUBLIC_URL}/images/loss_curve_log_scale.png` },
              ]
            },
          ],
        },
        {
          id: 'time_class_sampling',
          title: 'Sampling',
          content: [
            {
              type: 'paragraph',
              text: 'Sampling involves using both the time-step and class embeddings as input conditions. The model iteratively denoises the noisy input similar to the time-conditioned sampling, but with the predicted noise adjusted using classifier-free guidance (CFG). We used gamma = 5.0'
            },
            {
              type: 'math',
              text: '\\( \\hat{\\epsilon} = \\epsilon_u + \\gamma (\\epsilon_c - \\epsilon_u) \\)'
            },
            {
              type: 'paragraph',
              text: ' Here are some results for various Epochs.'
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 1 Output', imageUrl: `${process.env.PUBLIC_URL}/images/samples_epoch_1.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 5 Output', imageUrl: `${process.env.PUBLIC_URL}/images/samples_epoch_5.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 10 Output', imageUrl: `${process.env.PUBLIC_URL}/images/samples_epoch_10.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 15 Output', imageUrl: `${process.env.PUBLIC_URL}/images/samples_epoch_15.png` },
              ]
            },
            {
              type: 'image-grid',
              columns: 1,
              images: [
                { title: 'Epoch 20 Output', imageUrl: `${process.env.PUBLIC_URL}/images/samples_epoch_20.png` },
              ]
            },
          ],
        },        
      ],
    },
  ];

  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-gradient-to-b from-black via-gray-900 to-black text-gray-300">
      {/* Navbar */}
      <Navbar parentSections={parentSections} />

      {/* Main Content */}
      <div className="flex-1 px-4 sm:px-2 lg:px-8 py-8 md:ml-64 w-full">
        <h1 className="text-3xl sm:text-2xl lg:text-4xl font-bold text-center mb-12 text-white">
          Diffusion Models
        </h1>

        {/* Render Parent Sections */}
        {parentSections.map((parent) => (
          <ParentSection
            key={parent.id}
            id={parent.id}
            title={parent.title}
            sections={parent.sections}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
