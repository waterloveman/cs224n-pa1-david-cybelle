package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.langmodel.SmoothUnigramLanguageModel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A language model using smoothed bigram counts
 *
 * @author Cybelle Smith
 * @author David Edwards
 */
public class BigramLanguageModel implements LanguageModel {

  private static final String START = "<S>";
  private static final String STOP = "</S>";
  
  private Counter<String> wordCounter;
  private double total;
  private SmoothUnigramLanguageModel unigrams;


  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public BigramLanguageModel() {
    wordCounter = new Counter<String>();
    total = Double.NaN;
  }

  /**
   * Constructs a bigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public BigramLanguageModel(Collection<List<String>> sentences) {
    this();
    train(sentences);
  }


  // -----------------------------------------------------------------------

  /**
   * Constructs a bigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all bigrams (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public void train(Collection<List<String>> sentences) {
    wordCounter = new Counter<String>();
    for (List<String> sentence : sentences) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(STOP);
      wordCounter.incrementCount(START+" "+stoppedSentence.get(0),1.0);
      for (int i=0; i<stoppedSentence.size()-1; i++) {
	wordCounter.incrementCount(stoppedSentence.get(i) + " " +
				   stoppedSentence.get(i+1), 1.0);
      }
    }
    unigrams = new SmoothUnigramLanguageModel(sentences);
    total = wordCounter.totalCount();
  }


  // -----------------------------------------------------------------------

  private double getWordProbability(String word) {
    double count = wordCounter.getCount(word);
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


