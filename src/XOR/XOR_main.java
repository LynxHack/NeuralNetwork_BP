package XOR;

public abstract class XOR_main {


    public static void main(String[] args){
        System.out.println("Running NeuralNet");

        //Run these settings
        int argNumInputs = 2;
        int argNumHidden = 4;
        double argLearningRate = 0.2;
        double argMomentumTerm = 0.0;
        double argA = 0;
        double argB = 1;

        NeuralNet xor = new NeuralNet(argNumInputs, argNumHidden, argLearningRate, argMomentumTerm, argA, argB);

        double[] data1 = {0, 0};
        double[] data2 = {0, 1};
        double[] data3 = {1, 0};
        double[] data4 = {1, 1};

        while(true){
            System.out.println(xor.train(data1, 0 ));
            System.out.println(xor.train(data2, 1 ));
            System.out.println(xor.train(data3, 1 ));
            System.out.println(xor.train(data4, 0 ));
        }
    }
    /**
     * Constructor. (Cannot be declared in an interface, but your implementation will need one)
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only.
     * @param arbB Integer upper bound of sigmoid used by the output neuron only.

    public abstract NeuralNet (
    int argNumInputs,
    int argNumHidden,
    double argLearningRate,
    double argMomentumTerm,
    double argA,
    double argB );
     */

}