# VizWiz Vision Model Comparison for Accessibility

A study of state-of-the-art vision models on degraded images from the VizWiz dataset to improve accessibility tools for visually impaired individuals.

## Overview

This project evaluates how well modern vision models handle real-world image quality issues (blur, poor lighting, misframing) that are common in photos taken by visually impaired users. We compare multiple model architectures to identify which perform best on degraded images for visual question answering tasks.

## Motivation

The VizWiz dataset contains images captured by visually impaired individuals with challenging quality issues like:
- Blur and motion artifacts
- Poor lighting and low contrast
- Incorrect framing and occlusions

These degradations impact VQA performance, making it harder for assistive technologies to help users in daily tasks like reading labels or identifying objects.

## Project Goals

1. **Dataset Analysis**: Filter and analyze VizWiz images with quality issues (blur, darkness, poor framing)
2. **Model Comparison**: Evaluate multiple vision models:
   - Pure vision models: ViT, EfficientNet, CLIP
   - Vision-language models: BLIP-2, LLaVA, InstructBLIP
3. **Performance Evaluation**: Benchmark accuracy and robustness across different degradation types
4. **Insights**: Identify which architectures best serve accessibility applications

## Dataset

**VizWiz-VQA**: Visual question answering dataset with ~31,000 images from blind users
- Quality annotations: blur, darkness, framing issues
- Natural language questions and answers

## Models Evaluated

|  Vision Models  | ViT, EfficientNet, CLIP |
| Vision-Language |      BLIP-2, LLaVA      |

## Installation
```bash
