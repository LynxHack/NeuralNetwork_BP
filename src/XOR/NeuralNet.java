package XOR;

import java.util.*;

public class NeuralNet implements NeuralNetInterface{
    final double bias = 1.0;

    public double A;
    public double B;
    public double LearningRate;
    public double MomentumTerm;
    public int NumInputs;
    public int NumHidden;

    public double[][] weightInputToHidden = new double[NumHidden][NumInputs + 1];
    public double[][] previousWeightInputToHidden = new double[NumHidden][NumInputs + 1];

    public double[] weightHiddenToOutput;
    public double[] previousWeightHiddenToOutput;
    /**
     * Constructor for NeuralNet
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only.
     * @param argB Integer upper bound of sigmoid used by the output neuron only.
     */
    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA, double argB)
    {
        // Update our private variables
        A = argA;
        B = argB;

        LearningRate = argLearningRate;
        MomentumTerm = argMomentumTerm;

        NumInputs = argNumInputs;
        NumHidden = argNumHidden;

        weightInputToHidden = new double[NumHidden][NumInputs + 1];
        previousWeightInputToHidden = new double[NumHidden][NumInputs + 1];

        weightHiddenToOutput = new double[NumHidden + 1];
        previousWeightHiddenToOutput = new double[NumHidden + 1];
    }


    /**
     * Returns bipolar sigmoid for input x
     * @param x: Input
     * @return f(x) = 2 / (1+exp(-x)) - 1
     */
    public double sigmoid(double x){
        return 2 / (1 + Math.exp(-1 * x));
    }


    /**
     * Method implements general sigmoid asymtote bound (a,b)
     * @param x: Input
     * @return f(x) = b_minus_a / (1+exp(-x)) - minus_a
     */
    public double customSigmoid(double x){
        return (B - A) / ((1 + Math.exp(-1 * x)) - A);
    }

    /**
     * Initialize weights with for nodes (randomized)
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    public void initializeWeights(){
        // Initialize the weight InputToHidden to random values
        for (int i = 0; i < NumHidden; i++) {
            for (int j = 0; j < NumInputs + 1; j++) {
                weightInputToHidden[i][j] = Math.random() - 0.5;
            }
        }

        // Initialize the weight HiddenToOutput to random values
        for (int j = 0; j < NumHidden + 1; j++) {
            weightHiddenToOutput[j] = Math.random() - 0.5;
        }
    }
    /**
     * Initialize weights to 0
     */
    public void zeroWeights(){
        // Initialize the weight InputToHidden to random values
        for (int i = 0; i < NumHidden; i++) {
            for (int j = 0; j < NumInputs + 1; j++) {
                weightInputToHidden[i][j] = 0;
            }
        }

        // Initialize the weight HiddenToOutput to random values
        for (int j = 0; j < NumHidden + 1; j++) {
            weightHiddenToOutput[j] = 0;
        }
    }
}
