package XOR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface{
    final double bias = 1.0;

    public double a;
    public double b;
    public double b_minus_a;
    public double minus_a;
    public double LearningRate;
    public double MomentumTerm;
    public int numInputs;
    public int numHidden;

    public double[][] weightInputToHidden;
    public double[][] previousWeightInputToHidden;

    public double[] weightHiddenToOutput;
    public double[] previousWeightHiddenToOutput;

    private double [] hiddenNeuron; 					// hidden neuron vector

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

        LearningRate = argLearningRate;
        MomentumTerm = argMomentumTerm;

        numInputs = argNumInputs;
        numHidden = argNumHidden;

        weightInputToHidden = new double[numHidden][numInputs + 1];
        previousWeightInputToHidden = new double[numHidden][numInputs + 1];

        weightHiddenToOutput = new double[numHidden + 1];
        previousWeightHiddenToOutput = new double[numHidden + 1];

        hiddenNeuron = new double [numHidden];
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
        for ( int i = 0; i < numHidden; i++ ) {
            weightedSumHidden[i] = 0;
            for ( int j=0; j<numInputs; j++ ) {
                weightedSumHidden[i] += x[j] * weightInputToHidden[i][j];
            }
            weightedSumHidden[i] += bias * weightInputToHidden[i][numInputs];
            hiddenNeuron[i] = sigmoid( weightedSumHidden[i]);
        }

        // Compute weighted sum of output neuron
        weightedSumOutput = 0;
        for ( int i = 0; i < numHidden; i++ )
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
        double actualOutput = outputFor(x);
        double errorAtOutput = argvalue - actualOutput;

        // Derivative for a general sigmoid bounded by the range (a,b). See Fausett pg 309 for details.
        double sigmoidPrimeAtOutput = (1 / b_minus_a ) * (minus_a + actualOutput) * (b_minus_a - minus_a - actualOutput);
        double deltaAtOutput = errorAtOutput * sigmoidPrimeAtOutput; // delta is the error signal

        double weightChange = 0;

        if (argvalue > b) {
            System.out.println("*** NeuralNet: Target of " + argvalue + " exceeds upper bound of " + b);
        }
        if (argvalue < a) {
            System.out.println("*** NeuralNet: Target of " + argvalue + " exceeds lower bound of " + a);
        }

        return errorAtOutput;
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
