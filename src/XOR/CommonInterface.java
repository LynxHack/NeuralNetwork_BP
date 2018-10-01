package XOR;

import java.io.File;
import java.io.IOException;


public interface CommonInterface {
    /**
     *
     * @param x: The input Vector. double array.
     * @return Value returned by th LUT or NN for input vector
     */
    public double outputFor(double [] x);


    /**
     * Method tells NN or LUT output value to map to input vector
     * ex. The desired correct output value for given input
     * @param x: The input vector
     * @param argvalue: The new value to learn
     * @return The error of output for input vector
     */
    public double train(double [] x, double argvalue);


    /**
     * Write either LUT or weights to for the NN to a file
     * @param argFile: type file input
     */
    public void save(File argFile);

    /**
     * Loads LUT / NN weights from given file. Load have knowledge of how data
     * was written by save method. Raise error when trying to load incorrect format
     * ex. Wrong number of neurons
     * @param argFileName
     * @throws IOException
     */
    public void load(String argFileName) throws IOException;
}


