package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.lang.Math;
import java.lang.Integer;

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
	private Vector<int> nkhistogram;
	private double total;
	private double total_UNK_prob
	
	
	// -----------------------------------------------------------------------
	
	/**
	 * Constructs a new, empty unigram language model.
	 */
	public EmpiricalUnigramLanguageModel() {
		wordCounter = new Counter<String>();
		total = Double.NaN;
	}
	
	/**
	 * Constructs a unigram language model from a collection of sentences.  A
	 * special stop token is appended to each sentence, and then the
	 * frequencies of all words (including the stop token) over the whole
	 * collection of sentences are compiled.
	 */
	public EmpiricalUnigramLanguageModel(Collection<List<String>> sentences) {
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
		nkhistogram.set(0,0);
		int maxnk = 0;
		for (List<String> sentence : sentences) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(STOP);
			for (String word : stoppedSentence) {
				wordCounter.incrementCount(word, 1.0);
				int currCount = wordCounter.getCount(word);
				if (currCount > maxnk){
					maxnk = currCount;
					nkhistogram.set(currCount,0);
				}
				nkhistogram.set(currCount,++nkhistogram.get(currCount));
				nkhistogram.set(currCount-1,--nkhistogram.get(currCount));
			}
		}
		total = wordCounter.totalCount();
		adjustCounts();
	}
	
	/*
	 * Follow the procedure in Gale and Sampson 1995, Good-Turing Frequency Estimation without Tears, to calculate the adjusted unigram frequency counts.
	 */
    private void adjustCounts(){
		Vector<double> Z;
		Vector<double> logR;
		Vector<double> logZ;
		Vector<double> r_star;
		double Zvalue;
		double i;
		double k;
		double a;
		double b;
		double x;
		double y;
		
        for (int j = 1; j < nkhistogram.size(); j++){
			if (j == 1){
				i = 0;
			} else {
				i = j -1;
			}
			if (j==(nkhistogram.size() - 1)){
				k = 2j - i;
			} else {
				k = j + 1;
			}
			Zvalue = (2*nkhistogram.get(j))/(k-i);
			Z.set(j,Zvalue);
		}
		getLogarithms(nkhistogram.size()-1,logR);
		getLogarithms(Z,logR);
		setLeastSquaresParameters(a,b,logR,logZ);
		for (int r = 1; r < nkhistogram.size(); r++){
			x = (r + 1) * (nkhistogram.get(r + 1)/nkhistogram.get(r));
			y = (r + 1) * (S(a, b, r + 1) / S(a, b, r));
			if (useXnotY(x, y, r)){
				r_star.set(r, x);
			} else {
				r_star.set(r, y);
			}
		}
    }
	
    private void getLogarithms(int nRows, Vector<double> logs){
		for (int  i = 1; i < nRows+1; i++){
			logs.set(i,log(i.doubleValue()));
		}
    }
	
    private void getLogarithms(Vector<double> tobeLogged, Vector<double>
							   logs){
		for (int i = 1; i < tobeLogged.size(); i++){
			logs.set(i,log((tobeLogged.get(i)).doubleValue()));
		}
    }
	
	/**
	 * Sets the parameters a and b, such that logZ =  a + b*logR is the best fit
	 * to the pairs of values in the logR and logZ vectors.
	 */
    private void setLeastSquaresParameters(double a, double b,
					   Vector<double> logR,
					   Vector<double> logZ){
		double logR_Mean = getMean(logR);
		double logZ_Mean = getMean(logZ);
		double b_numerator = 0;
		double b_denominator = 0;
		for (int row = 1; row < logR.size(); row++){
			b_numerator += (logR.get(row) -	logR_Mean)*(logZ.get(row) - 
								    logZ_Mean);
			b_denominator += pow((logR.get(row) - logR_Mean),2.0);
		}
		b = b_numerator / b_denominator;
		a = logZ_Mean - (b * logR_Mean);
    }
	
    private double getMean(Vector<double> values){
		// since the zeroith index  doesn't count for our purposes
		double n_values = values.size() - 1;
		double total;
		double mean;
		for (int i = 1; i < values.size(); i++){
			total += values.get(i);
		}
		mean = total / n_values;
		return mean;
    }
	
    private double S(double a, double b, int  r){
		return exp(a + (b * log(r)));
    }
	
    private boolean useXnotY(double x, double y, int r){
		double n_rplusone = (double) nkhistogram.get(r+1);
		double n_r = (double) nkhistogram.get(r);
		double rplusone = (double) (r + 1);
		double crazyEquation = 1.96 * sqrt( pow(rplusone, 2)*(n_rplusone/pow(n_r,2))*(1 + (n_rplusone/ n_r);
		if (abs(x - y)>(crazyEquation)){
			return true;
		} else {
			return false;
		}
	}
							 
							 
// -----------------------------------------------------------------------
							 
private double getWordProbability(String word) { double count = wordCounter.getCount(word);
	 if (count == 0) {                   // unknown word
					    // System.out.println("UNKNOWN WORD: " + sentence.get(index));
	     return 1.0 / (total + 1.0);
	 }
	 return count / (total + 1.0);
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
								 double probability = 1.0;
								 for (int index = 0; index < stoppedSentence.size(); index++) {
									 probability *= getWordProbability(stoppedSentence, index);
								 }
								 return probability;
							 }
							 
							 /**
							  * checks if the probability distribution properly sums up to 1
							  */
							 public double checkModel() {
								 double sum = 0.0;
								 // since this is a unigram model, 
								 // the event space is everything in the vocabulary (including STOP)
								 // and a UNK token
								 
								 // this loop goes through the vocabulary (which includes STOP)
								 for (String word : wordCounter.keySet()) {
									 sum += getWordProbability(word);
								 }
								 
								 // remember to add the UNK. In this EmpiricalUnigramLanguageModel
								 // we assume there is only one UNK, so we add...
								 sum += 1.0 / (total + 1.0);
								 
								 return sum;
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
									 sum += wordCounter.getCount(word) / total;
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
								 while (!word.equals(STOP)) {
									 sentence.add(word);
									 word = generateWord();
								 }
								 return sentence;
							 }
							 
							 }