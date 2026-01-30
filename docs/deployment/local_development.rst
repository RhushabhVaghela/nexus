.. _local_development:

Local Development Setup
=======================

This guide covers setting up Nexus for local development on your machine.

Prerequisites
-------------

* Python 3.10 or higher
* CUDA-capable GPU with 8GB+ VRAM (recommended)
* 16GB+ system RAM
* Git

Installation Steps
------------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/yourusername/nexus.git
   cd nexus

2. Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using conda (recommended):

.. code-block:: bash

   conda create -n nexus python=3.10
   conda activate nexus

Or using venv:

.. code-block:: bash

   python -m venv nexus-env
   source nexus-env/bin/activate  # On Windows: nexus-env\Scripts\activate

3. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

4. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   pytest tests/ -v

Running Specific Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profile a teacher model:

.. code-block:: bash

   python scripts/niwt_profiler.py --model codellama/CodeLlama-7b-hf

Run the full pipeline:

.. code-block:: bash

   ./run_nexus_master.sh --models "coder" --datasets "code_search_net"

VSCode Configuration
--------------------

Recommended extensions:

* Python
* Pylance
* Jupyter
* Markdownlint
* Even Better TOML

Create ``.vscode/settings.json``:

.. code-block:: json

   {
       "python.defaultInterpreterPath": "./nexus-env/bin/python",
       "python.linting.enabled": true,
       "python.linting.pylintEnabled": true,
       "python.formatting.provider": "black"
   }

Troubleshooting
---------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

Reduce batch size or use gradient checkpointing:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0
   python scripts/train.py --batch_size 1 --gradient_checkpointing

Import Errors
~~~~~~~~~~~~~

Ensure the src directory is in your Python path:

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
