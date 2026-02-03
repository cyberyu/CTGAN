Based on the sources provided, yes, you can use a nearest neighbor lookup strategy to "borrow" data from Clients 1 and 2 to mimic Client 3.
However, to do this effectively using the provided library, you should not use the standard CTGAN model. Instead, you should use the TVAE (Tabular Variational AutoEncoder) model, which is included in the same codebase,.
Here is why and how to implement this "Latent Space Nearest Neighbor" strategy:
Why Standard CTGAN Cannot Do This
The standard CTGAN architecture is designed to generate data from random noise (z∼N(0,I)) and a conditional vector,. It is a "one-way" generator (Noise→Data). It lacks a mechanism to take real data (Client 3) and map it backward to understand its features or find similar points in the generator's space.
The Solution: Use TVAE for Latent Space Lookup
The TVAE model is the correct tool for this task because it contains an Encoder network,. The Encoder can map complex, mixed-type tabular rows into a compressed numerical vector (Latent Vector z). This allows you to perform "Nearest Neighbor" lookups in a semantically meaningful space rather than trying to match raw, messy tabular data.
Step-by-Step Implementation
1. Train on Allowable Data (Client 1 & 2) Train the TVAE model using only the data from Client 1 and Client 2.
• Input: T_train (Client 1 + Client 2).
• Process: The model learns to compress this data into a latent space (mean μ and variance σ) and reconstruct it.
• Outcome: The model now understands the "grammar" of the allowable data.
2. Project All Data into Latent Space Once trained, use the Encoder (inference mode) to project the data into latent vectors. This does not update the model weights, satisfying your constraint that Client 3 is not used for training.
• Encode Client 1 & 2: Pass the allowable data through the Encoder to get a pool of allowable latent vectors (Z 
allowable
​
 ).
• Encode Client 3: Pass Client 3's data through the same Encoder to get target latent vectors (Z 
target
​
 ),.
3. Perform Nearest Neighbor Lookup Perform the lookup in this latent space (Z) rather than the raw data space. This is superior because the TVAE's preprocessing handles the complexity of mixed discrete/continuous columns and mode-specific normalization,.
• Action: For each vector in Z 
target
​
  (Client 3), find the closest vector in Z 
allowable
​
  (Client 1 & 2) using Euclidean distance.
• Selection: Select the actual rows from Client 1 and Client 2 that correspond to these nearest neighbors.
4. Result: The "Mimic" Dataset You now have a dataset composed entirely of rows from Client 1 and Client 2 (which is allowed), but the distribution of these rows has been selected to statistically mimic Client 3.
Summary of Differences for this Task
Feature
CTGAN Model
TVAE Model
Architecture
Generator + Discriminator (GAN)
Encoder + Decoder (VAE)
Input Capability
Takes Random Noise + Conditions
Takes Real Data (to encode)
Can it "Look up"?
No. Cannot map real data to internal representation.
Yes. Can map Client 3 data to Latent Space z.
Best Use Case
Generating entirely new synthetic samples
projecting data or creating variations
Code Concept (Using TVAE from the CTGAN library)
from ctgan import TVAE

# 1. Train on Allowable Data
tvae = TVAE()
tvae.fit(client_1_and_2_data, discrete_columns)

# 2. Access the DataTransformer and Encoder (Internal components)
# Note: You would typically access the internal Pytorch module 
# to run the 'transform' and 'encoder' steps manually.
transformer = tvae._transformer
encoder = tvae._generator.encoder

# 3. Transform and Encode Data (Conceptual)
# Convert raw rows to neural-net ready format
c1_c2_processed = transformer.transform(client_1_and_2_data)
c3_processed = transformer.transform(client_3_data)

# Get Latent Vectors (Z)
z_allowable = encoder(c1_c2_processed) 
z_target = encoder(c3_processed)

# 4. Nearest Neighbor (using standard library like sklearn)
# Find which z_allowable is closest to each z_target
# Return the original rows from client_1_and_2_data