import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import plot


sentences = [
    # From July 8, 2017 New York Times:
    'Wissenschaftler des CERN-Labors sagen, sie hätten ein neues Teilchen entdeckt.',
    'Es gibt eine Möglichkeit, die akute emotionale Intelligenz zu messen, die nie aus der Mode gekommen ist.',
    'Präsident Trump traf sich mit anderen Führern auf der Konferenz der 20-köpfigen Gruppe.',
    'Der Gesetzentwurf des Senats zur Aufhebung und Ersetzung des Affordable Care Act ist nun gefährdet.',
    # From Google's Tacotron example page:
    'Generatives kontradiktorisches Netzwerk oder Variations-Auto-Encoder.',
    'Die Busse sind nicht das Problem, sie bieten tatsächlich eine Lösung.',
    'Spricht der schnelle braune Fuchs über den faulen Hund?',
    'Talib Kweli bestätigte gegenüber AllHipHop, dass er im nächsten Jahr ein Album veröffentlichen wird.',
    # From mycroft
    "Ich brauchte eine ziemlich lange Zeit, um eine Stimme zu entwickeln, und jetzt, da ich sie habe, werde ich nicht schweigen.",
    "Sei eine Stimme, kein Echo.",
    "Die menschliche Stimme ist das vollkommenste Instrument von allen.",
    "Es tut mir leid, Dave, ich fürchte, das kann ich nicht.",
    "This cake is great, It's so delicious and frisch.",
    "Hallo, mein Name ist Mycroft.",
    "hallo.",
    "Beeindruckend.",
    "cool.",
    "großartig.",
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    wav_path = '%s-%d.wav' % (base_path, i)
    align_path = '%s-%d.png' % (base_path, i)
    print('Synthesizing and plotting: %s' % wav_path)
    wav, alignment = synth.synthesize(text)
    with open(wav_path, 'wb') as f:
      f.write(wav)
    plot.plot_alignment(
        alignment, align_path,
        info='%s' % (text)
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint', required=True,
      help='Path to model checkpoint')
  parser.add_argument(
      '--hparams', default='',
      help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument(
      '--force_cpu', default=False,
      help='Force synthesize with cpu')
  parser.add_argument(
      '--gpu_assignment', default='0',
      help='Set the gpu the model should run on')

  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment

  if args.force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
