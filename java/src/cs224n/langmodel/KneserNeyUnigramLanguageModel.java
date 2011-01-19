package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Vector;
import java.lang.Math.*;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class KneserNeyUnigramLanguageModel implements LanguageModel {

  private static final String STOP = "</S>";
  
  private Counter<String> wordCounter;
  private Vector nkhistogram;
  private double total;
  private double delta;
	private double V;
	private double C;
	private double NO;


  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public KneserNeyUnigramLanguageModel() {
    wordCounter = new Counter<String>();
	nkhistogram = new Vector();
    total = Double.NaN;
	  V = Double.NaN;
	  C = Double.NaN;
	  NO = Double.NaN;
	  delta = Double.NaN;
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public KneserNeyUnigramLanguageModel(Collection<List<String>> sentences) {
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
	double wordCount;
	double nWordsWithCurrNK;
	for (int i = 0; i < 3; i++){
		nkhistogram.add(0.0);
	}
	for (String word : wordCounter.keySet()) {
		wordCount = wordCounter.getCount(word);
		if (wordCount == 1 || wordCount == 2){
			nWordsWithCurrNK = ((Double) nkhistogram.get((int) wordCount)).doubleValue() + 1.0;
			nkhistogram.set((int) wordCount,  nWordsWithCurrNK);
		}
	}
    total = wordCounter.totalCount();
	V = wordCounter.size();
	NO = 1;
	C = total;
	  double NOne = ((Double) nkhistogram.get(1)).doubleValue();
	  double NTwo = ((Double) nkhistogram.get(2)).doubleValue();
	  delta = (NOne / (NOne + (2 * NTwo)));
  }


  // -----------------------------------------------------------------------

  private double getWordProbability(String word) {
	  
    double count = wordCounter.getCount(word);
	if (!wordCounter.containsKey(word) || count == 0){              // unknown word
		return (0.0 - (Math.log(delta * (V - NO)) - Math.log(NO * C)));
      // System.out.println("UNKNOWN WORD: " + sentence.get(index));
    }
    return (0.0 - (Math.log(count - delta) - Math.log(C)));
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
      probability += getWordProbability(stoppedSentence, index);
    }
	  //System.out.println("Sentence Probability: " + Math.exp(-probability));
    return Math.exp(0.0 - probability);
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
      sum += Math.exp(0.0 - getWordProbability(word));
    }
    
    // remember to add the UNK. In this EmpiricalUnigramLanguageModel
    // we assume there is only one UNK, so we add...
	  System.out.println("delta: " + delta);
	  System.out.println("V: " + V);
	  System.out.println("NO: " + NO);
	  System.out.println("C: " + C);
	  System.out.println("Minus Log UNK probability: " + -(Math.log(delta * (V - NO)) - Math.log(NO*C)));
    sum += Math.exp(Math.log(delta * (V - NO)) - Math.log(NO*C));
    
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


