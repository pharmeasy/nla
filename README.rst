.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style

.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity
   :alt: Maintenance

.. image:: https://img.shields.io/badge/python-3.above-blue.svg
   :target: https://img.shields.io/badge/python-3.above-blue.svg
   :alt: versions


Natural Language Augmentor (NLA)
=================================
As Emmitt Smith would say, *"All models are created equal. Some work harder in training."*

Augmentation, a critical step in the model training pipeline, is a way to get the model
ready for the hardships of the real world. It distorts the smooth training track to
simulate the rocky road. Thus, the more realistic the augmentation, the more realistic
the training data and therefore, sturdier the model.

NLA, Natural Language Augmentor, is the ultimate real and fast way to augment
textual data for use in any model training. It has all the means to create natural
and random textual errors.

The *homophones* module replaces any word with its homophones.
The *keyaug* and *randaug* modules introduce lightning-fast and parallelized streams
of organised (keyboard sensitive) and random errors. The *edge n-gram* and *word boundary*
modules introduce n-gram deletions and word boundary errors respectively.

NLA is a one-stop-shop for all lexical augmentations of any text.

Installation
============
Use the package manager pip to install *nla*.
Replace 'branch-name' with the name of the branch you want to pull.

.. code-block:: sh

    pip install git+ssh://git@github.com:ahwspl/nla.git@branch-name


Usage
=====

There are some arguments that the below functions take
which might need some explanation.

**degree**

- This is an int/float argument which refers to the intensity of
  the augmentation.

**count**

- This is the tentative number of outputs you want per query.

**parallel**

- This is a boolean argument that asks if you want to run
  augmentation on a large data in parallel.

**method**

- *method*, is the method of augmentation. It takes five values,
  'swap', 'insert', 'replace', 'delete' and 'random'.

**position**

- *position*, takes four values, 'first',
  'middle', 'end' and 'random'. This argument refers
  to the position in the string which we want to augment.

****kwargs**

- This is an identifier argument. This is what you use if you need
  to set some identifier in the output of the augmentation.
  The examples below will makes things more clear.



Homophones
------------
Generates homophones for a word.
 

.. code-block:: python
    
    from nla.homophones import *
    
    """
    :param query: word to augment
    :type query: str
    :param beamwidth: number of outputs
    :type count: int
    :return: list of homophones of the input word
    """

    QUERY = 'CARROM'
    genome(QUERY, beamwidth=20)

    >> ['CARAM',
        'KARUM',
        'CAROM',
        'CAREM',
        'KHARROM',
        'KAREM',
        'KARAM',
        'CARROME']


    QUERY = "POWDER"
    genome(QUERY, beamwidth=5)

    >> ['POUDER',
        'PAUDER',
        'POWEDER',
        'POODER',
        'PAWDAR']

Nearest Neighbour Keyboard Augmentation
----------------------------------------
This module mimics the typing errors from a QWERTY keyboard.

The code below works at a word level. It takes a list of words as input.


.. code-block:: python

    from nla.keyboard.keyaug import *

    """
    run the given augmentation on given list of words
    :param words: list of words to augment
    :type words: list
    :param kwargs:
    :return:
    """

    query = ["DEEP", "NEURAL", "CRAVING"]
    nn_fetch(
        words=query,
        degree=2,
        count=2,
        method="insert",
        position="first",
        parallel=True,
        identifier="generic",
    )

    >> [('DSEEP', 'DEEP', 'generic'),
        ('DREEP', 'DEEP', 'generic'),
        ('NSEIURAL', 'NEURAL', 'generic'),
        ('NSEJURAL', 'NEURAL', 'generic'),
        ('CDRAVING', 'CRAVING', 'generic'),
        ('CTRAVING', 'CRAVING', 'generic')]



The code below works at a sentence level. It takes a list of sentences as input.

.. code-block:: python

    from nla.keyboard.nlaug import keyboard_sent_aug

    """
    run the given augmentation on a sentence
    :param sentence: sentence to augment
    :type sentence: str
    :return:
    """

    data = ["DEEP NEURAL", "CRAVING MANIFEST"]
    keyboard_sent_aug(
        sentences=data,
        degree=2,
        count=2,
        method="random",
        position="random",
        parallel=True,
        dummy_identifier_1="generic1",
        dummy_identifier_2="generic2",
    )

    >> [['FDEELP NWURQL', 'DEEP NEURAL', 'generic1', 'generic2'],
        ['DSSP HEURQL', 'DEEP NEURAL', 'generic1', 'generic2'],
        ['CRAVJIHNG MANIGFESRT', 'CRAVING MANIFEST', 'generic1', 'generic2'],
        ['CRAWVINHG MANJFRST', 'CRAVING MANIFEST', 'generic1', 'generic2']]



Random Character Augmentation
------------------------------
The code below works at a word level. It takes a list of words as input.

.. code-block:: python

    from nla.keyboard.randaug import *

    """
    run the given augmentation on given list of words
    :param words: list of words to augment
    :type words: list
    :param kwargs:
    :return:
    """

    query = ["DEEP", "NEURAL", "CRAVING"]
    fetch(
        words=query,
        degree=2,
        count=2,
        method="random",
        position="random",
        parallel=True,
        dummy_identifier_1="generic1",
        dummy_identifier_2="generic2"
    )

    >> [('DEP', 'DEEP', 'generic1', 'generic2'),
        ('DGEFEP', 'DEEP', 'generic1', 'generic2'),
        ('NUERAL', 'NEURAL', 'generic1', 'generic2'),
        ('NEUJRAL', 'NEURAL', 'generic1', 'generic2'),
        ('CDRSAVING', 'CRAVING', 'generic1', 'generic2'),
        ('CRTAVING', 'CRAVING', 'generic1', 'generic2')]



The code below works at a sentence level. It takes a list of sentences.

.. code-block:: python

    from nla.keyboard.nlaug import rand_sent_aug

    """
    run the given random augmentation on a sentence
    :param sentence: sentence to augment
    :type sentence: str
    :return:
    """

    data = ["DEEP", "NEURAL", "CRAVING"]
    rsa = rand_sent_aug(
        sentences=data,
        degree=2,
        count=2,
        method="random",
        position="random",
        parallel=True,
        dummy_identifier_1="generic1",
        dummy_identifier_2="generic2",
    )

    >> [['DEP', 'DEEP', 'generic1', 'generic2'],
        ['DVEJEP', 'DEEP', 'generic1', 'generic2'],
        ['NKMRAL', 'NEURAL', 'generic1', 'generic2'],
        ['NEUORTAL', 'NEURAL', 'generic1', 'generic2'],
        ['CARVIGN', 'CRAVING', 'generic1', 'generic2'],
        ['CRAVIYNDG', 'CRAVING', 'generic1', 'generic2']]



Edge N-gram
----------------
Takes list of sentences.


.. code-block:: python

    from nla.edge_n_gram import *

    """
    :param queries: sentences to augment
    :type queries: list
    :param kwargs: identifiers
    :return: list of tuple with augmented sentence, original sentence and identifiers
    """

    data = ["DEEP NEURAL", "CRAVING MANIFEST"]

    edge_n_gram(
            queries=data,
            count=2,
            degree=2,
            parallel=False,
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2",
        )

    >> [[('DEE NEUR', 'DEEP NEURAL', 'generic1', 'generic2'),
        ('DEEP NEURA', 'DEEP NEURAL', 'generic1', 'generic2')],
        [('CRAVIN MANIFE', 'CRAVING MANIFEST', 'generic1', 'generic2'),
        ('CRAVI MANIFE', 'CRAVING MANIFEST', 'generic1', 'generic2')]]



Word Boundary
----------------
Takes list of sentences.


.. code-block:: python

   from nla.word_boundary import *

   """
    :param queries: sentences to augment
    :type queries: list
    :param kwargs: identifiers
    :return: list of tuple with augmented sentence, original sentence and identifiers
    """

   data = ["DEEP NEURAL", "CRAVING MANIFEST"]

   word_boundary(
            queries=data,
            count=2,
            degree=0.6,
            parallel=True,
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2",
        )

   >> [('DEEPNEURAL', 'DEEP NEURAL', 'generic1', 'generic2'),
       ('DEEP NE U R AL', 'DEEP NEURAL', 'generic1', 'generic2'),
       ('DEE PNE URAL', 'DEEP NEURAL', 'generic1', 'generic2'),
       ('M AN IFEST CR AVIN G', 'CRAVING MANIFEST', 'generic1', 'generic2'),
       ('MANIF ESTCRAV ING', 'CRAVING MANIFEST', 'generic1', 'generic2'),
       ('MANIFESTCRAVING', 'CRAVING MANIFEST', 'generic1', 'generic2')]
