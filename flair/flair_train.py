from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.visual.training_curves import Plotter

#train over the 1.3 UDT dataset

columns = {1: 'text', 2: 'lemma', 3: 'pos', 4:'morph'}
corpus: Corpus = ColumnCorpus('', columns,
                                train_file='lv-ud-train13.conllu',
                                test_file='lv-ud-test211.conllu')
label_type = 'pos' #var būt vajadzīgs nomainīt

label_dict = corpus.make_label_dictionary(label_type = label_type)
print(corpus)
embedding_types = [
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = SequenceTagger(
    hidden_size = 256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type=label_type,
    use_crf=True
)

trainer = ModelTrainer(tagger,corpus)
path = 'models13/pos/'
trainer.train(path,
   learning_rate=0.1,
   mini_batch_size=16,
   max_epochs=200,
   checkpoint=True,
   embeddings_storage_mode='cpu',
   write_weights=True
)

plotter = Plotter()
plotter.plot_training_curves(path+'loss.tsv')
plotter.plot_weights(path+'weights.txt')