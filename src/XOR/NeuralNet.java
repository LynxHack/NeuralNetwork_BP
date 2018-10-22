package XOR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

public class NeuralNet implements NeuralNetInterface{
    public double bias = 1.0;

    public double a;
    public double b;
    public double b_minus_a;
    public double minus_a;
    public double learningRate;
    public double momentumTerm;
    public int numInputs;
    public int numHidden;

    public double[][] weightInputToHidden;
    public double[][] previousWeightInputToHidden;

    public double[] weightHiddenToOutput;
    public double[] previousWeightHiddenToOutput;

    public double [] hiddenNeuron; 					// hidden neuron vector


    public double[][] inputValues;
    public double[] targetValues;
    public double tmpOut = 0;
    public double outputError = 0;
    public double[] hiddenError;

    // temp store current weight, used in momentum
    public double[][] tempWeightInputToHidden;
    public double[] tempWeightHiddenToOutput;
    public double[] hiddenValues;


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
        a = argA;
        b = argB;
        b_minus_a = b - a;
        minus_a = -1 * a;

        learningRate = argLearningRate;
        momentumTerm = argMomentumTerm;

        numInputs = argNumInputs;
        numHidden = argNumHidden;

        weightInputToHidden = new double[numHidden][numInputs + 1];
        previousWeightInputToHidden = new double[numHidden][numInputs + 1];

        weightHiddenToOutput = new double[numHidden + 1];
        previousWeightHiddenToOutput = new double[numHidden + 1];


        // initialize hidden layer
        hiddenNeuron = new double [numHidden];
        hiddenValues = new double[numHidden];
        inputValues = new double[numHidden][numInputs + 1];
        targetValues = new double[numHidden];
        hiddenError = new double[numHidden];


        // temp store current weight, used in momentum
        tempWeightInputToHidden = new double[numHidden][numInputs + 1];
        tempWeightHiddenToOutput = new double[numHidden + 1];


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
        return (b_minus_a) / ((1 + Math.exp(-1 * x)) - minus_a);
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
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInputs + 1; j++) {
                weightInputToHidden[i][j] = Math.random() - 0.5;
            }
        }

        // Initialize the weight HiddenToOutput to random values
        for (int j = 0; j < numHidden + 1; j++) {
            weightHiddenToOutput[j] = Math.random() - 0.5;
        }
    }
    /**
     * Initialize weights to 0
     */
    public void zeroWeights(){
        // Initialize the weight InputToHidden to random values
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInputs + 1; j++) {
                weightInputToHidden[i][j] = 0;
            }
        }

        // Initialize the weight HiddenToOutput to random values
        for (int j = 0; j < numHidden + 1; j++) {
            weightHiddenToOutput[j] = 0;
        }
    }

    /**
     *
     * @param x: The input Vector. double array.
     * @return Value returned by th LUT or NN for input vector
     */
    public double outputFor(double [] x){

        double [] weightedSumHidden = new double[numHidden];
        double weightedSumOutput;

        if ( x.length != numInputs ) {
            System.out.println("-** Length of input vector expected: " + numInputs + "Got: " + x.length);
            return 0;
        }

        // Compute weighted sum at hidden neurons
        for (int i = 0; i < numHidden; i++){
            weightedSumHidden[i] = 0;
            for (int j = 0; j < numInputs; j++){
                weightedSumHidden[i] += x[j] * weightInputToHidden[i][j];
            }
            weightedSumHidden[i] += bias * weightInputToHidden[i][numInputs];
            hiddenNeuron[i] = sigmoid(weightedSumHidden[i]);
        }

        // Compute weighted sum of output neuron
        weightedSumOutput = 0;
        for (int i = 0; i < numHidden; i++)
            weightedSumOutput += hiddenNeuron[i] * weightHiddenToOutput[i];
        weightedSumOutput += bias * weightHiddenToOutput[numHidden];

        // We have the final output. Return it.
        return customSigmoid(weightedSumOutput);

    }


    /**
     * Method tells NN or LUT output value to map to input vector
     * ex. The desired correct output value for given input
     * @param x: The input vector
     * @param argvalue: The new value to learn
     * @return The error of output for input vector
     */
    public double train(double [] x, double argvalue){
        // feed forward
        calcHiddenValue(x, argvalue);
        calcOutputValue();

        // backPrapogation
        calcOutputError(argvalue);
        calcHiddenError();

        // update weights
        updateWeightInputToHidden(x);
        updateWeightHiddenToOutput();
        return 0;
    }

    private void calcHiddenValue(double[] X, double target) {
        for (int i = 0; i < numHidden; i++) {
            hiddenValues[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                hiddenValues[i] += X[j] * weightInputToHidden[i][j];
            }
            hiddenValues[i] += weightInputToHidden[i][numInputs] * 1; // bias

            // processed by activation function
            hiddenValues[i] = sigmoid(hiddenValues[i]);
        }
    }

    private void calcOutputValue() {
        double tmpOut = 0;
        for (int i = 0; i < numHidden; i++) {
            tmpOut += hiddenValues[i] * weightHiddenToOutput[i];
        }
        tmpOut += weightHiddenToOutput[numHidden] * 1; // bias
        tmpOut = sigmoid(tmpOut); // activation function
    }

    private void calcOutputError(double target) {
        // for bipolar input
        outputError = 0.5 * (1 +tmpOut) * (1 - tmpOut) * (target - tmpOut);

        // for binary input
        //outputError = tmpOut * (1 - tmpOut) * (target - tmpOut);
        //System.out.println("output error:" + outputError + "\ttmpOut: " + tmpOut + "\ttarget: " + target);
    }

    private void calcHiddenError() {
        // for bipolar input
        for (int i = 0; i < numHidden; i++) {
            hiddenError[i] = 0.5 * (1 +hiddenValues[i]) * (1 - hiddenValues[i])
                    * outputError * weightHiddenToOutput[i];

            // for binary input
//		for (int i = 0; i < numHidden; i++) {
//			hiddenError[i] = hiddenValues[i] * (1 - hiddenValues[i])
//					* outputError * weightHiddenToOutput[i];
            //System.out.println(i + " hidden error" + hiddenError[i]);
        }
    }

    private void updateWeightInputToHidden(double[] X) {
        // // backup current weight
        for (int i = 0; i < numHidden; i++) {
            System.arraycopy(weightInputToHidden[i], 0, tempWeightInputToHidden[i], 0,
                    weightInputToHidden[i].length);
        }

        // update weightInputToHidden
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInputs; j++) {
                weightInputToHidden[i][j] += learningRate
                        * hiddenError[i] * X[j] + momentumTerm
                        * getPreviousToHiddenDeltaWeight(i, j);
                //System.out.println("numHidden: " + i + "  Inputs: " + X[j] + "  weight: " +weightInputToHidden[i][j]);
            }
            // for bias term below
            weightInputToHidden[i][numInputs] += learningRate
                    * hiddenError[i] + momentumTerm
                    * getPreviousToHiddenDeltaWeight(i, numInputs);
            //System.out.println("numHidden: " + i + "  Inputs: " +  X[numInputs] + "  weight: " +weightInputToHidden[i][numInputs]);
        }

        // update previous weight
        for (int i = 0; i < numHidden; i++) {
            System.arraycopy(tempWeightInputToHidden[i], 0, previousWeightInputToHidden[i], 0,
                    weightInputToHidden[i].length);
        }

    }

    private void updateWeightHiddenToOutput() {
        // backup current weight
        System.arraycopy(weightHiddenToOutput, 0, tempWeightHiddenToOutput, 0,
                weightHiddenToOutput.length);

        for (int i = 0; i < numHidden; i++) {
            weightHiddenToOutput[i] += learningRate * outputError
                    * hiddenValues[i] + momentumTerm
                    * getPreviousToOutDeltaWeight(i);
            //System.out.println("numHidden: " + i + "  weight: " +weightHiddenToOutput[i]);
        }
        weightHiddenToOutput[numHidden] += learningRate * outputError * 1
                + momentumTerm * getPreviousToOutDeltaWeight(numHidden);
        // update previous weight
        System.arraycopy(tempWeightHiddenToOutput, 0,
                previousWeightHiddenToOutput, 0,
                tempWeightHiddenToOutput.length);
    }

    private double getPreviousToOutDeltaWeight(int i) {
        if (previousWeightHiddenToOutput[i] != 0)
            return weightHiddenToOutput[i] - previousWeightHiddenToOutput[i];
        else
            return 0;
    }

    private double getPreviousToHiddenDeltaWeight(int i, int j) {
        if (previousWeightInputToHidden[i][j] != 0)
            return weightInputToHidden[i][j]
                    - previousWeightInputToHidden[i][j];
        else
            return 0;
    }
    /**
     * Write either LUT or weights to for the NN to a file
     * @param argFile: type file input
     */
    public void save(File argFile){
        PrintStream saveFile = null;

        try {
            saveFile = new PrintStream( new FileOutputStream( argFile ));
        }
        catch (IOException e) {
            System.out.println( "*** Could not create output stream for NN save file.");
        }

        saveFile.println(numInputs);
        saveFile.println(numHidden);

        // First save the weights from the input to hidden neurons (one line per weight)
        for ( int i=0; i<numHidden; i++) {
            for ( int j=0; j<numInputs; j++) {
                saveFile.println( weightInputToHidden [i][j] );
            }
            saveFile.println(weightInputToHidden [i][numInputs]); // Save bias weight for this hidden neuron too
        }
        // Now save the weights from the hidden to the output neuron
        for (int i=0; i<numHidden; i++) {
            saveFile.println(weightHiddenToOutput[i]);
        }
        saveFile.println(weightHiddenToOutput[numHidden]); // Save bias weight for output neuron too.
        saveFile.close();
    }

    /**
     * Loads LUT / NN weights from given file. Load have knowledge of how data
     * was written by save method. Raise error when trying to load incorrect format
     * ex. Wrong number of neurons
     * @param argFileName
     * @throws IOException
     */
    public void load(String argFileName) throws IOException{

        FileInputStream inputFile = new FileInputStream( argFileName );
        BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));

        // Check that NN defined for file matches that created
        int numInputInFile = Integer.valueOf( inputReader.readLine() );
        int numHiddenInFile = Integer.valueOf( inputReader.readLine() );

        if (numInputInFile != numInputs) {
            System.out.println ( "*** Number of inputs in file is " + numInputInFile + " Expected " + numInputs );
            throw new IOException();
        }
        if (numHiddenInFile != numHidden) {
            System.out.println ( "*** Number of hidden in file is " + numHiddenInFile + " Expected " + numHidden );
            throw new IOException();
        }
        if ((numInputInFile != numInputs) || (numHiddenInFile != numHidden)) {
            return;
        }

        // First load the weights from the input to hidden neurons (one line per weight)
        for ( int i=0; i<numHidden; i++) {
            for ( int j=0; j<numInputs; j++) {
                weightInputToHidden [i][j] = Double.valueOf( inputReader.readLine() );
            }
            weightInputToHidden [i][numInputs] = Double.valueOf( inputReader.readLine() ); // Load bias weight for this hidden neuron too
        }
        // Now load the weights from the hidden to the output neuron
        for (int i=0; i<numHidden; i++) {
            weightHiddenToOutput [i] = Double.valueOf( inputReader.readLine() );
        }
        weightHiddenToOutput [numHidden] = Double.valueOf( inputReader.readLine() ); // Load bias weight for output neuron too.

        inputReader.close();
    }
}
