package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Vector;
import java.util.Random;
import java.lang.Math.*;
import java.lang.Integer;
import java.lang.Double;


/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class SmoothUnigramLanguageModel implements LanguageModel {
	
	
	private static final String STOP = "</S>";
	
	private Counter<String> wordCounter;
	private Vector nkhistogram;
	private Vector unigramLogProbability;
	private double total;

	
	
	// -----------------------------------------------------------------------
	
	/**
	 * Constructs a new, empty unigram language model.
	 */
	public SmoothUnigramLanguageModel() {
		wordCounter = new Counter<String>();
		nkhistogram = new Vector();
		unigramLogProbability = new Vector();
		total = Double.NaN;
	}
	
	/**
	 * Constructs a unigram language model from a collection of sentences.  A
	 * special stop token is appended to each sentence, and then the
	 * frequencies of all words (including the stop token) over the whole
	 * collection of sentences are compiled.
	 */
	public SmoothUnigramLanguageModel(Collection<List<String>> sentences) {
		this();
		train(sentences);
	}
	
	
	// -----------------------------------------------------------------------
	
	/**
	 * Constructs a unigram language model from a collection of sentences.  A
	 * special stop token is appended to each sentence, and then the
	 * frequencies of all words (including the stop token) over the whole
	 * collection of sentences are compiled.
	 */
	public void train(Collection<List<String>> sentences) {
		wordCounter = new Counter<String>();
		
		for (List<String> sentence : sentences) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(STOP);
			for (String word : stoppedSentence) {
				wordCounter.incrementCount(word, 1.0);
			}
		}
		total = wordCounter.totalCount();
		double wordCount;
		double nWordsWithCurrNK;
		nkhistogram.add(0.0);
		for (String word : wordCounter.keySet()) {
			wordCount = wordCounter.getCount(word);
			//System.out.println(word + "     " + wordCount);
			while (wordCount > nkhistogram.size() - 1){
				nkhistogram.add(0.0);
			}
			nWordsWithCurrNK = ((Double) nkhistogram.get((int) wordCount)).doubleValue() + 1.0;
			nkhistogram.set((int) wordCount,  nWordsWithCurrNK);
		}
		System.out.println("number of words with count " + 165 + " is: " + nkhistogram.get(165));
		/*for (int i = 0; i < nkhistogram.size(); i++){
			System.out.println("number of words with count " + i + " is: " + nkhistogram.get(i));
		}*/
		
		adjustCounts();
	}
	
	/*
	 * Follow the procedure in Gale and Sampson 1995, Good-Turing Frequency Estimation without Tears, to calculate the adjusted unigram frequency counts.
	 */
    private void adjustCounts(){
		Vector Z = new Vector();
		Vector logR = new Vector();
		Vector logZ = new Vector();
		Vector r_star = new Vector();
		double Zvalue;
		double i;
		double k;
		double a = 0.0;
		double b = 0.0;
		double x;
		double y;
		double N_dash = 0.0;
		
		for (int rep = 0; rep < nkhistogram.size(); rep++){
			Z.add(0.0);
			logR.add(0.0);
			logZ.add(0.0);
			r_star.add(0,0);
		}
		
        for (int j = 1; j < nkhistogram.size(); j++){
        	if (((Double) nkhistogram.get(j)).doubleValue() != 0.0){
        		if (j == 1){
        			i = 0;
        		} else {
        			i = j -1;
        		}
        		if (j==(nkhistogram.size() - 1)){
        			k = (2*j - i);
        		} else {
        			k = j + 1;
        		}
        		Zvalue = (2*((Double) nkhistogram.get(j)).doubleValue())/(k-i);
        		Z.set(j,Zvalue);
        	}
		}
		getLogarithms(logR);
		getLogarithms(Z,logR);
		setLeastSquaresParameters(a,b,logR,logZ);
	
		int rPlusOne;
		for (int r = 1; r < nkhistogram.size(); r++){
			if (((Double) nkhistogram.get(r)).doubleValue() != 0.0){
				rPlusOne = getNextR(r);
				y = rPlusOne * (S(a, b, r + 1) / S(a, b, r));
				if (rPlusOne < nkhistogram.size()){
					x = rPlusOne * (((Double) nkhistogram.get(rPlusOne)).doubleValue()/((Double) nkhistogram.get(r)).doubleValue());
					if (useXnotY(x, y, r)){
						r_star.set(r, x);
						System.out.println("X  r_star of row " + r + " is: " + x);
					} else {
						r_star.set(r, y);
						System.out.println("Y  r_star of row " + r + " is: " + y);
					}
				} else {
					r_star.set(r, y);
				}	
				N_dash = N_dash + ((Double) nkhistogram.get(r)).doubleValue() * ((Double) r_star.get(r)).doubleValue();
			}
		}
		unigramLogProbability.setSize(nkhistogram.size());
		System.out.println("Number of words with count 1: " + nkhistogram.get(1));
		unigramLogProbability.set(0, -Math.log((((Double) nkhistogram.get(1)).doubleValue()/(total))));
		System.out.println("UNK probability: " + Math.exp(-((Double)unigramLogProbability.get(0)).doubleValue()));
		for (int r = 1; r < nkhistogram.size(); r++){
			if (((Double) nkhistogram.get(r)).doubleValue() != 0.0){
				unigramLogProbability.set(r, -Math.log((1.0 -(((Double) nkhistogram.get(1)).doubleValue()/total))*(((Double) r_star.get(r)).doubleValue()/N_dash)));
				System.out.println("ulp of words with count " + r + " is: " + unigramLogProbability.get(r));
			}
		}
		/*double normalizedCountProb;
		for (int r = 50; r < nkhistogram.size(); r++){
		    normalizedCountProb = Math.log(((Double) nkhistogram.get(r)).doubleValue()) - Math.log(total + 1);
		    unigramLogProbability.set(r, normalizedCountProb);
		    System.out.println("ulp of words with count " + r + " is: " + unigramLogProbability.get(r));
		}*/
    }
    
    private int getNextR(int r){
    	int j = r + 1;
    	while (j < nkhistogram.size()){
    		if (((Double) nkhistogram.get(r)).doubleValue() != 0.0) break;
    		j++;
    	}
    	return j;
    }
	
    private void getLogarithms(Vector logs){
		for (int  i = 1; i < nkhistogram.size(); i++){
			logs.set(i,java.lang.Math.log((double) i));
		}
    }
	
    private void getLogarithms(Vector tobeLogged, Vector
							   logs){
		for (int i = 1; i < tobeLogged.size(); i++){
			logs.set(i,java.lang.Math.log(((Double) tobeLogged.get(i)).doubleValue()));
		}
    }
	
	/**
	 * Sets the parameters a and b, such that logZ =  a + b*logR is the best fit
	 * to the pairs of values in the logR and logZ vectors.
	 */
    private void setLeastSquaresParameters(double a, double b,
					   Vector logR,
					   Vector logZ){
		double logR_Mean = getMean(logR);
		double logZ_Mean = getMean(logZ);
		double b_numerator = 0;
		double b_denominator = 0;
		for (int row = 1; row < logR.size(); row++){
			b_numerator = b_numerator + (((Double) logR.get(row)).doubleValue() -	logR_Mean)*(((Double) logZ.get(row)).doubleValue() - 
								    logZ_Mean);
			b_denominator = b_denominator + java.lang.Math.pow((((Double) logR.get(row)).doubleValue() - logR_Mean),2.0);
		}
		b = b_numerator / b_denominator;
		a = logZ_Mean - (b * logR_Mean);
    }
	
    private double getMean(Vector values){
		// since the zeroith index  doesn't count for our purposes
		double sum = 0.0;
		double mean;
		for (int i = 1; i < values.size(); i++){
			sum = sum + ((Double) values.get(i)).doubleValue();
		}
		mean = sum / wordCounter.size();
		return mean;
    }
	
    private double S(double a, double b, int  r){
		return java.lang.Math.exp(a + (b * java.lang.Math.log((double) r)));
    }
	
    private boolean useXnotY(double x, double y, int r){
    	int nextR = getNextR(r);
		double n_rplusone = ((Double) nkhistogram.get(nextR)).doubleValue();
		double n_r = ((Double) nkhistogram.get(r)).doubleValue();
		double rplusone = (double) nextR;
		double crazyEquation = 1.96 * java.lang.Math.sqrt(java.lang.Math.pow(rplusone, 2)*(n_rplusone/java.lang.Math.pow(n_r,2))*(1 + (n_rplusone/ n_r)));
		if (java.lang.Math.abs(x - y)> crazyEquation ){
			return true;
		} else {
			return false;
		}
	}
							 
							 
// -----------------------------------------------------------------------
							 
private double getWordProbability(String word) { 
	 int count;
	 double logProb;
	 double unkProb = ((Double) unigramLogProbability.get(0)).doubleValue();
	 if (wordCounter.containsKey(word)){
		 count = (int) wordCounter.getCount(word);
		 logProb = ((Double) unigramLogProbability.get(count)).doubleValue();
		 //System.out.println("returned -logProb " + logProb + " for word: " + word + " which has count " + wordCounter.getCount(word));
		 if (logProb == java.lang.Double.POSITIVE_INFINITY){
			 System.out.println("Error: word " + word + " has -logProb positive infinity");
			 //return 13;
		 } 
		 return logProb;
	 } else {
		 return unkProb;
	 }
}
							 
							 /**
							  * Returns the probability, according to the model, of the word specified
							  * by the argument sentence and index.  Smoothing is used, so that all
							  * words get positive probability, even if they have not been seen
							  * before.
							  */
							 public double getWordProbability(List<String> sentence, int index) {
								 String word = sentence.get(index);
								 return getWordProbability(word);
							 }
							 
							 /**
							  * Returns the probability, according to the model, of the specified
							  * sentence.  This is the product of the probabilities of each word in
							  * the sentence (including a final stop token).
							  */
							 public double getSentenceProbability(List<String> sentence) {
								 List<String> stoppedSentence = new ArrayList<String>(sentence);
								 stoppedSentence.add(STOP);
								 double probability = 0.0;
								 for (int index = 0; index < stoppedSentence.size(); index++) {
									 probability = probability + getWordProbability(stoppedSentence, index);
								 }
								 return (probability);
							 }
							 
							 /**
							  * checks if the probability distribution properly sums up to 1
							  */
							 public double checkModel() {
								 double sum = 0.0;
								 for (String word : wordCounter.keySet()) {
									 sum = sum + Math.exp(-getWordProbability(word));
								 }
								 double unkProb = Math.exp(-((Double) unigramLogProbability.get(0)).doubleValue());
								 sum = sum + unkProb;
								 return sum;
								 
								 /*double sum = 0.0;
								 // since this is a unigram model, 
								 // the event space is everything in the vocabulary (including STOP)
								 // and a UNK token
								 
								 // this loop goes through the vocabulary (which includes STOP)
								 double numerator = 0.0;
								 double wordProb;
								 for (String word : wordCounter.keySet()) {
									 wordProb = getWordProbability(word);
									 sum = sum + wordProb;
									 numerator += Math.exp(wordProb);
								 }
								 
								 // remember to add the UNK. In this EmpiricalUnigramLanguageModel
								 double unkProb = ((Double) unigramLogProbability.get(0)).doubleValue();
								 sum = sum + unkProb;
								 numerator += Math.exp(unkProb);
								
								 System.out.println("numerator in Check model: " + numerator);
								 System.out.println("denominator in Check model: " + Math.exp(sum));
								 double modelProb = (numerator / Math.exp(sum));
								 return modelProb;*/
							 }
							 
							 /**
							  * Returns a random word sampled according to the model.  A simple
							  * "roulette-wheel" approach is used: first we generate a sample uniform
							  * on [0, 1]; then we step through the vocabulary eating up probability
							  * mass until we reach our sample.
							  */
							 public String generateWord() {
								 double sample = Math.random();
								 double sum = 0.0;
								 for (String word : wordCounter.keySet()) {
									 sum = sum + Math.exp(-getWordProbability(word));
									 if (sum > sample) {
										 return word;
									 }
								 }
								 return "*UNKNOWN*";   // a little probability mass was reserved for unknowns
							 }
							 
							 /**
							  * Returns a random sentence sampled according to the model.  We generate
							  * words until the stop token is generated, and return the concatenation.
							  */
							 public List<String> generateSentence() {
								 List<String> sentence = new ArrayList<String>();
								 String word = generateWord();
								 //while (!word.equals(STOP)) {
								 for (int i = 0; i < 10; i++){
									 sentence.add(word);
									 word = generateWord();
								 }
								 //}
								 return sentence;
							 }
							 
					}