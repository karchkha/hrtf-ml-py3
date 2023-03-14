# Neural Networks

File structure

	main-network.py 
        initializer.py: read arguments and setups up defaults
        utilities/parameters.py: holds
                - the informatiom about the different databases
                - training orders
                - training parameters
    network_classes
        different networks combined in main-network
    utilities
        different utilities used in the network creating and training


## Network Structure


```mermaid
graph TD

    In[Position, Anthro]-->A;
    In-->B;
    In-->C;
    In-->D;
    In-->E;

    A[Real, Realmean, Realstd]-->D[MagRI];
    B[Imag, Imagmean, Imagstd]-->D[MagRI];
    
    C[Mag, Magmean, Magstd]-->|Mag| E[MagFinal];
    D[MagRI]-->E[MagFinal];
    
    E[MagFinal]-->G(Reconstruct);
    
    C-.->|Magmean, Magstd| G(MagFinal Reconstruct);
    D-.-> I(MagRI Reconstruct);
    C-.->|Magmean, Magstd| I(MagRI Reconstruct);
    C-.-> H(Mag Reconstruct);
    A-.-> J(Real Reconstruct);
    B-.-> K(Imag Reconstruct);
    
    style A fill:#FEE
    style B fill:#FEE
    style C fill:#FEE
    
    style G fill:#EFF
    style H fill:#EFF
    style I fill:#EFF
    style J fill:#EFF
    style K fill:#EFF

```

## Data Formatting
  * The data is stratified by utilities/network_data.py into the following categories and percentages
    * 10% Test Data
    * 72% training data
    * 18% validatiaon data (20% of 90%)

  * All random stratifications are seeded
    * Test data is stratified with "0"
    * and training and validation with "iteration+100"
