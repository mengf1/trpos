#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;

// configurable hyper parameters
float pdrop = 0.5;
unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 0;

// set dynamically from input files
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE_cl = 0;
unsigned EMBEDDING_DIM = 0;

// map a word to an embedding using cross-lingual word embeddings
std::map<int, vector<float>> embeddings_cl;

bool eval = false;

// dictionary for all languages
dynet::Dict d;
dynet::Dict td;

int kNONE;
int kSOS;
int kEOS;
int UNK;

// use the universal tagset
//const string TAG_SET[] = {"VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."};

template <class Builder>
struct UniTagger
{
  LookupParameter p_w_cl;

  Parameter p_l2th;
  Parameter p_r2th;
  Parameter p_thbias;

  Parameter p_th2t;
  Parameter p_tbias;
  Builder l2rbuilder;
  Builder r2lbuilder;
  explicit UniTagger(Model &model)
      : l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
        r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
  {
    // do not need to generate word embeddings
    p_w_cl = model.add_lookup_parameters(VOCAB_SIZE_cl, {INPUT_DIM});
    // NOW SET THEIR VALUES!
    // iterate over map<int, floats>
    // copy floats into p_w[i] for each word index i in map
    for (std::map<int, vector<float>>::iterator iter = embeddings_cl.begin(); iter != embeddings_cl.end(); iter++)
    {
      p_w_cl.initialize(iter->first, iter->second);
    }

    p_l2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_r2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

    p_th2t = model.add_parameters({TAG_SIZE, TAG_HIDDEN_DIM});
    p_tbias = model.add_parameters({TAG_SIZE});
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const vector<int> &sent, const vector<int> &tags,
                               ComputationGraph &cg, double *cor = 0,
                               unsigned *ntagged = 0)
  {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg); // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg); // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias);
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    // read sequence from left to right
    l2rbuilder.add_input(const_lookup(cg, p_w_cl, kSOS));

    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = const_lookup(cg, p_w_cl, sent[t]);
      if (!eval)
      {
        i_words[t] = noise(i_words[t], 0.1);
      }
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(const_lookup(cg, p_w_cl, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    for (unsigned t = 0; t < slen; ++t)
    {
      if (tags[t] != kNONE)
      {
        if (ntagged)
          (*ntagged)++;
        Expression i_th = tanh(
            affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
        if (!eval)
        {
          i_th = dropout(i_th, pdrop);
        }
        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
        if (cor)
        {
          vector<float> dist = as_vector(cg.incremental_forward(i_t));
          double best = -9e99;
          int besti = -1;
          for (int i = 0; i < dist.size(); ++i)
          {
            if (dist[i] > best)
            {
              best = dist[i];
              besti = i;
            }
          }
          if (tags[t] == besti)
            (*cor)++;
        }
        if (tags[t] != kNONE)
        {
          Expression i_err = pickneglogsoftmax(i_t, tags[t]);
          errs.push_back(i_err);
        }
      }
    }
    return sum(errs);
  }

  // prediction
  void PredictTags(const vector<int> &sent, const vector<int> &tags,
                   ComputationGraph &cg,
                   unsigned *ntagged = 0, std::ofstream &outputFile = nullptr)
  {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg); // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg); // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias);
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    // read sequence from left to right
    l2rbuilder.add_input(const_lookup(cg, p_w_cl, kSOS));

    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = const_lookup(cg, p_w_cl, sent[t]);
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(const_lookup(cg, p_w_cl, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    for (unsigned t = 0; t < slen; ++t)
    {
      if (tags[t] == kNONE)
      {
        if (ntagged)
          (*ntagged)++;
        Expression i_th = tanh(
            affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});

        vector<float> dist = as_vector(cg.incremental_forward(i_t));
        double best = -9e99;
        int besti = -1;
        for (int i = 0; i < dist.size(); ++i)
        {
          if (dist[i] > best)
          {
            best = dist[i];
            besti = i;
          }
        }

        //save the tags
        outputFile << td.convert(besti) << " ";
      }
    }
    outputFile << "\n";
  }
};

vector<string> split(const string &s, char c)
{
  vector<string> parts;
  string::size_type i = 0;
  string::size_type j = s.find(c);

  while (j != string::npos)
  {
    parts.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);

    if (j == string::npos)
      parts.push_back(s.substr(i, s.length()));
  }
  return parts;
}

std::map<int, vector<float>> importEmbeddings(string inputFile)
{
  std::map<int, vector<float>> w_embeddings;
  unsigned welc = 0;
  unsigned dim = 0;
  cerr << "Loading word embeddings " << inputFile << "...\n";

  string line;
  ifstream in(inputFile);
  assert(in);

  while (getline(in, line))
  {

    // word, (val1, val2, ...)
    // if use multiCCA, then the space is ' '
    vector<string> splitedline = split(line, ' ');

    string word = splitedline[0];
    auto w_id = d.convert(word);
    vector<float> w_embedding;

    for (unsigned i = 1; i < splitedline.size(); i++)
    {
      w_embedding.push_back(std::stof(splitedline[i]));
    }
    if (dim == 0)
      dim = splitedline.size() - 1;
    else
      assert(dim == splitedline.size() - 1);
    w_embeddings[w_id] = w_embedding;

    ++welc;
  }

  cerr << welc << " word embeddings, " << dim << " dimensions \n";
  return w_embeddings;
}

vector<pair<vector<int>, vector<int>>> loadCorpus(string inputFile)
{
  cerr << "Loading data " << inputFile << "...\n";
  vector<pair<vector<int>, vector<int>>> data;
  string line;
  int lc = 0;
  int toks = 0;
  ifstream in(inputFile);
  assert(in);

  while (getline(in, line))
  {
    ++lc;
    int nc = 0;
    vector<int> x, y;
    read_sentence_pair(line, x, d, y, td);
    assert(x.size() == y.size());
    if (x.size() == 0)
    {
      cerr << line << endl;
      abort();
    }
    data.push_back(make_pair(x, y));
    for (unsigned i = 0; i < y.size(); ++i)
    {
      if (y[i] != kNONE)
      {
        ++nc;
      }
    }
    if (nc == 0)
    {
      cerr << "No tagged tokens in line " << lc << endl;
      abort();
    }
    toks += x.size();
  }
  cerr << lc << " lines, " << toks << " tokens, " << d.size() << " types\n";
  cerr << "Tags: " << td.size() << endl;
  return data;
}

void trainData(vector<pair<vector<int>, vector<int>>> &training, UniTagger<LSTMBuilder> &tagger, Trainer *sgd)
{
  int report = 0;
  int report_every_i = 100;
  double loss = 0;
  double cor = 0;
  unsigned ttags = 0;
  unsigned budget = training.size();
  cerr << "\nTraining labelled data: size = " << budget << " sentences";
  for (unsigned i = 0; i < budget; ++i)
  {
    // build graph for this instance
    ComputationGraph cg;
    auto &sent = training[i];
    //cerr << "Compute loss\n";
    Expression loss_expr = tagger.BuildTaggingGraph(sent.first, sent.second, cg, &cor, &ttags);
    loss += as_scalar(cg.forward(loss_expr));
    cg.backward(loss_expr);
    sgd->update(1.0);
    ++report;
    if (report % report_every_i == 0)
    {
      cerr << "\n***TRAIN: E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (cor / ttags) << ") ";
    }
  }
}

void evaluateData(vector<pair<vector<int>, vector<int>>> &test, UniTagger<LSTMBuilder> &tagger, string resultFile)
{

  ofstream outputW;
  outputW.open(resultFile);

  //test data
  double loss = 0;
  unsigned ttags = 0;
  eval = true;
  for (auto &sent : test)
  {
    ComputationGraph cg;
    tagger.PredictTags(sent.first, sent.second, cg, &ttags, outputW);
  }
  eval = false;
  outputW.close();
  cerr << "\n***TEST  Done ";
}

//argv: embeddings training save_model max_epoch
int main(int argc, char **argv)
{
  dynet::initialize(argc, argv);

  kNONE = td.convert("*");
  kSOS = d.convert("<S>");
  kEOS = d.convert("</S>");

  vector<pair<vector<int>, vector<int>>> training, dev;
  string line;

  // load cross-lingual word embeddings
  unsigned welc = 0;
  unsigned dim = 0;
  string embeddingFile = argv[1];
  embeddings_cl = importEmbeddings(embeddingFile);
  d.freeze(); // no new word types allowed
  d.set_unk("<UNK>");
  VOCAB_SIZE_cl = d.size();

  string dataFile = argv[2];
  training = loadCorpus(dataFile);
  td.freeze();
  TAG_SIZE = td.size();

  // initialise the model
  ostringstream os;
  os << "tagger"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  bool use_momentum = true;
  Trainer *sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(model);
  else
    sgd = new SimpleSGDTrainer(model);

  UniTagger<LSTMBuilder> tagger(model);
  int MAX_EPOCH = 10;
  if (argc == 4)
  {
    MAX_EPOCH = atoi(argv[4]);
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;

  int epoch = 0;
  while (1)
  {
    Timer iteration("completed in");
    double loss = 0;
    //epoch++;
    cerr << "\n***Training [epoch=" << epoch << "]";
    // train labelled data and unlabelled data
    trainData(training, tagger, sgd);

    cerr << "\n***Best = " << best;
    sgd->status();
    sgd->update_epoch();
    epoch++;
    if (epoch > MAX_EPOCH)
      break;
  }

  //save model

  ofstream out(fname);
  boost::archive::text_oarchive oa(out);
  oa << model;

  delete sgd;
}
