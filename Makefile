# Makefile for Memorization Experiments
#
# Author: G.J.J. van den Burg
# Copyright (c) 2021, The Alan Turing Institute
# License: See the LICENSE file.
#

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

SCRIPT_DIR=./scripts
SUMMARY_DIR=./summaries
OUTPUT_DIR=./output
RESULT_DIR=./results
PAPER_IMG_DIR=../../../paper/images/

RESULT_DIR_BMNIST_LR3=$(RESULT_DIR)/binarized_mnist_extra_lr3/results
RESULT_DIR_BMNIST_LR4=$(RESULT_DIR)/binarized_mnist_extra_lr4/results
RESULT_DIR_CIFAR10=$(RESULT_DIR)/cifar10/results
RESULT_DIR_CELEBA=$(RESULT_DIR)/celeba/results

CKPT_DIR_BMNIST_LR3=$(RESULT_DIR)/binarized_mnist_extra_lr3/checkpoints
CKPT_DIR_BMNIST_LR4=$(RESULT_DIR)/binarized_mnist_extra_lr4/checkpoints
CKPT_DIR_CIFAR10=$(RESULT_DIR)/cifar10/checkpoints
CKPT_DIR_CELEBA=$(RESULT_DIR)/celeba/checkpoints

# dependencies of memorization.py
MEM_FILES=\
	  $(SCRIPT_DIR)/constants.py \
	  $(SCRIPT_DIR)/dataset.py \
	  $(SCRIPT_DIR)/models.py \
	  $(SCRIPT_DIR)/trainer.py \
	  $(SCRIPT_DIR)/seed_generator.py

##########
#        #
# GLOBAL #
#        #
##########

.PHONY: all

all: memorization flow_lr_comparison summaries analysis

############################
#                          #
# MEMORIZATION EXPERIMENTS #
#                          #
############################

.PHONY: memorization

memorization: \
	mem_mnist_lr3 \
	mem_mnist_lr4 \
	mem_cifar10 \
	mem_celeba

############################
# BinarizedMNIST lr = 1e-3 #
############################

BMNIST_LR3_CV_TARGETS=
BMNIST_LR3_FULL_TARGETS=

define make_targets_cv_bmnist_lr3
BMNIST_LR3_CV_TARGETS += $(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_bmnist_lr3
BMNIST_LR3_FULL_TARGETS += $(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_bmnist_lr3,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_bmnist_lr3,42,0))

.PHONY: mem_mnist_lr3 mem_mnist_cv_lr3 mem_mnist_full_lr3

mem_mnist_lr3: mem_mnist_cv_lr3 mem_mnist_full_lr3

mem_mnist_cv_lr3: $(BMNIST_LR3_CV_TARGETS)

mem_mnist_full_lr3: $(BMNIST_LR3_FULL_TARGETS)

.PRECIOUS: $(BMNIST_LR3_FULL_TARGETS) $(BMNIST_LR3_CV_TARGETS)

$(BMNIST_LR3_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR3) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR3)

$(BMNIST_LR3_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR3) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR3)

############################
# BinarizedMNIST lr = 1e-4 #
############################

BMNIST_LR4_CV_TARGETS=
BMNIST_LR4_FULL_TARGETS=

define make_targets_cv_bmnist_lr4
BMNIST_LR4_CV_TARGETS += $(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_bmnist_lr4
BMNIST_LR4_FULL_TARGETS += $(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_bmnist_lr4,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_bmnist_lr4,42,0))

.PHONY: mem_mnist_lr4 mem_mnist_cv_lr4 mem_mnist_full_lr4

mem_mnist_lr4: mem_mnist_cv_lr4 mem_mnist_full_lr4

mem_mnist_cv_lr4: $(BMNIST_LR4_CV_TARGETS)

mem_mnist_full_lr4: $(BMNIST_LR4_FULL_TARGETS)

.PRECIOUS: $(BMNIST_LR4_FULL_TARGETS) $(BMNIST_LR4_CV_TARGETS)

$(BMNIST_LR4_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR4) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR4)

$(BMNIST_LR4_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR4) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR4)

############
# CIFAR-10 #
############

CIFAR10_CV_TARGETS=
CIFAR10_FULL_TARGETS=

define make_targets_cv_cifar10
CIFAR10_CV_TARGETS += $(RESULT_DIR_CIFAR10)/CIFAR10_DiagonalGaussianDCVAE_NF32-L64_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_cifar10
CIFAR10_FULL_TARGETS += $(RESULT_DIR_CIFAR10)/CIFAR10_DiagonalGaussianDCVAE_NF32-L64_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_cifar10,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_cifar10,42,0))

.PHONY: mem_cifar10 mem_cifar10_cv mem_cifar10_full

mem_cifar10: mem_cifar10_cv mem_cifar10_full

mem_cifar10_cv: $(CIFAR10_CV_TARGETS)

mem_cifar10_full: $(CIFAR10_FULL_TARGETS)

.PRECIOUS: $(CIFAR10_FULL_TARGETS) $(CIFAR10_CV_TARGETS)

$(CIFAR10_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CIFAR10 \
		--model DiagonalGaussianDCVAE \
		--mode split-cv \
		--latent-dim 64 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 50 \
		--result-dir $(RESULT_DIR_CIFAR10) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_CIFAR10)

$(CIFAR10_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CIFAR10 \
		--model DiagonalGaussianDCVAE \
		--mode full \
		--latent-dim 64 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 50 \
		--result-dir $(RESULT_DIR_CIFAR10) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_CIFAR10)
##########
# CelebA #
##########

CELEBA_CV_TARGETS=
CELEBA_FULL_TARGETS=

define make_targets_cv_celeba
CELEBA_CV_TARGETS += $(RESULT_DIR_CELEBA)/CelebA_ConstantGaussianDCVAE_NF32-L32_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_celeba
CELEBA_FULL_TARGETS += $(RESULT_DIR_CELEBA)/CelebA_ConstantGaussianDCVAE_NF32-L32_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_celeba,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_celeba,42,0))

.PHONY: mem_celeba mem_celeba_cv mem_celeba_full

mem_celeba: mem_celeba_cv mem_celeba_full

mem_celeba_cv: $(CELEBA_CV_TARGETS)

mem_celeba_full: $(CELEBA_FULL_TARGETS)

.PRECIOUS: $(CELEBA_FULL_TARGETS) $(CELEBA_CV_TARGETS)

$(CELEBA_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CelebA \
		--model ConstantGaussianDCVAE \
		--mode split-cv \
		--latent-dim 32 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 50 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 25 \
		--result-dir $(RESULT_DIR_CELEBA) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_CELEBA)

$(CELEBA_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CelebA \
		--model ConstantGaussianDCVAE \
		--mode full \
		--latent-dim 32 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 50 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 25 \
		--result-dir $(RESULT_DIR_CELEBA) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_CELEBA)


##################################
# Flow Learning Rate Comparison  #
##################################

# BinarizedMNIST with SimpleRealNVP - Learning Rate 1e-3
RESULT_DIR_FLOW_BMNIST_LR3=$(RESULT_DIR)/flow_bmnist_lr3/results
CKPT_DIR_FLOW_BMNIST_LR3=$(RESULT_DIR)/flow_bmnist_lr3/checkpoints

FLOW_BMNIST_LR3_CV_TARGETS=
FLOW_BMNIST_LR3_FULL_TARGETS=

define make_targets_cv_flow_bmnist_lr3
FLOW_BMNIST_LR3_CV_TARGETS += $(RESULT_DIR_FLOW_BMNIST_LR3)/BinarizedMNIST_SimpleRealNVP_L4-H256_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_flow_bmnist_lr3
FLOW_BMNIST_LR3_FULL_TARGETS += $(RESULT_DIR_FLOW_BMNIST_LR3)/BinarizedMNIST_SimpleRealNVP_L4-H256_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 4),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_flow_bmnist_lr3,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_flow_bmnist_lr3,42,0))

# BinarizedMNIST with SimpleRealNVP - Learning Rate 1e-4  
RESULT_DIR_FLOW_BMNIST_LR4=$(RESULT_DIR)/flow_bmnist_lr4/results
CKPT_DIR_FLOW_BMNIST_LR4=$(RESULT_DIR)/flow_bmnist_lr4/checkpoints

FLOW_BMNIST_LR4_CV_TARGETS=
FLOW_BMNIST_LR4_FULL_TARGETS=

define make_targets_cv_flow_bmnist_lr4
FLOW_BMNIST_LR4_CV_TARGETS += $(RESULT_DIR_FLOW_BMNIST_LR4)/BinarizedMNIST_SimpleRealNVP_L4-H256_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_flow_bmnist_lr4
FLOW_BMNIST_LR4_FULL_TARGETS += $(RESULT_DIR_FLOW_BMNIST_LR4)/BinarizedMNIST_SimpleRealNVP_L4-H256_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 4),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_flow_bmnist_lr4,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_flow_bmnist_lr4,42,0))

.PHONY: flow_lr_comparison mem_flow_bmnist_lr3 mem_flow_bmnist_lr3_cv mem_flow_bmnist_lr3_full
.PHONY: mem_flow_bmnist_lr4 mem_flow_bmnist_lr4_cv mem_flow_bmnist_lr4_full

# Main target: run both learning rate experiments
flow_lr_comparison: mem_flow_bmnist_lr3 mem_flow_bmnist_lr4

mem_flow_bmnist_lr3: mem_flow_bmnist_lr3_cv mem_flow_bmnist_lr3_full
mem_flow_bmnist_lr4: mem_flow_bmnist_lr4_cv mem_flow_bmnist_lr4_full

mem_flow_bmnist_lr3_cv: $(FLOW_BMNIST_LR3_CV_TARGETS)
mem_flow_bmnist_lr3_full: $(FLOW_BMNIST_LR3_FULL_TARGETS)

mem_flow_bmnist_lr4_cv: $(FLOW_BMNIST_LR4_CV_TARGETS)
mem_flow_bmnist_lr4_full: $(FLOW_BMNIST_LR4_FULL_TARGETS)

.PRECIOUS: $(FLOW_BMNIST_LR3_FULL_TARGETS) $(FLOW_BMNIST_LR3_CV_TARGETS)
.PRECIOUS: $(FLOW_BMNIST_LR4_FULL_TARGETS) $(FLOW_BMNIST_LR4_CV_TARGETS)

$(FLOW_BMNIST_LR3_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model SimpleRealNVP \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 20 \
		--repeats 5 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_FLOW_BMNIST_LR3) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_FLOW_BMNIST_LR3)

$(FLOW_BMNIST_LR3_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model SimpleRealNVP \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 50 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_FLOW_BMNIST_LR3) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_FLOW_BMNIST_LR3)

$(FLOW_BMNIST_LR4_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model SimpleRealNVP \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 20 \
		--repeats 5 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_FLOW_BMNIST_LR4) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_FLOW_BMNIST_LR4)

$(FLOW_BMNIST_LR4_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model SimpleRealNVP \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 50 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_FLOW_BMNIST_LR4) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_FLOW_BMNIST_LR4)

#################################
# Flow Model Summarization      #
#################################

# Create summary directory if it doesn't exist
summary-dir:
	mkdir -p $(SUMMARY_DIR)

# Learning rate comparison summaries
$(SUMMARY_DIR)/mem_flow_bmnist_lr3.npz: $(SCRIPT_DIR)/summarize.py \
	$(FLOW_BMNIST_LR3_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(FLOW_BMNIST_LR3_CV_TARGETS)

$(SUMMARY_DIR)/mem_flow_bmnist_lr4.npz: $(SCRIPT_DIR)/summarize.py \
	$(FLOW_BMNIST_LR4_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(FLOW_BMNIST_LR4_CV_TARGETS)

# Analysis targets for flow learning rate comparison
.PHONY: summaries analysis flow_lr_analysis

summaries: flow_lr_analysis

analysis: flow_lr_analysis

flow_lr_analysis: \
	$(SUMMARY_DIR)/mem_flow_bmnist_lr3.npz \
	$(SUMMARY_DIR)/mem_flow_bmnist_lr4.npz

