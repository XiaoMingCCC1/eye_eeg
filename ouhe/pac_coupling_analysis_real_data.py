#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-Amplitude Coupling (PAC) Analysis for Real EEG Data
Real Experimental Data - Batch Processing

This script analyzes phase-amplitude coupling between pupil size and EEG signals
across all available subjects and videos, generating coupling strength values.
"""

import numpy as np
import pandas as pd
import os
import glob
import re
from datetime import datetime
from scipy import stats
from scipy.signal import hilbert, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class PACAnalysisBatch:
    def __init__(self):
        """Initialize PAC analysis for batch processing"""
        self.eeg_dir = 'F:/eye_eeg/jiequ/jingxitai'
        self.pupil_dir = 'F:/eye_eeg/jiequ/jingxitai_yandong/shuju'
        self.results_dir = 'F:/eye_eeg/ouhe/results_jingxi'
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Analysis parameters
        self.sampling_rate = 500  # Hz
        
        print(f"🧠 PAC Analysis initialized for batch processing")
        print(f"📁 EEG data directory: {self.eeg_dir}")
        print(f"👁️ Pupil data directory: {self.pupil_dir}")
        print(f"📊 Results will be saved to: {self.results_dir}")
    
    def find_data_pairs(self):
        eeg_file = 'F:/eye_eeg/jiequ/jingxitai/SPRiNT_parameters_jingxitai_20251215_155258.log'
        pupil_file = 'F:/eye_eeg/jiequ/jingxitai_yandong/shuju/pupil_size_data_A00018886_Video1_SMP_101s_20251026_220726.log'
        print(f"🔍 Using explicit EEG file: {os.path.basename(eeg_file)}")
        print(f"🔍 Using explicit pupil file: {os.path.basename(pupil_file)}")
        basename = os.path.basename(pupil_file)
        match = re.match(r'pupil_size_data_([A-Z0-9]+)_Video(\d+)_.*\.log', basename)
        if match:
            subject_id = match.group(1)
            video_num = match.group(2)
        else:
            subject_id = 'UNKNOWN'
            video_num = '1'
        data_pairs = [{
            'subject_id': subject_id,
            'video_id': video_num,
            'eeg_file': eeg_file,
            'pupil_file': pupil_file
        }]
        print(f"\n📊 Total data pairs found: {len(data_pairs)}")
        return data_pairs
    
    def extract_subject_video_from_eeg_content(self, eeg_file):
        """Extract subject ID and video number from EEG file content"""
        try:
            with open(eeg_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Look for the line with "文件: " which contains the original filename
                for line in lines[:10]:  # Check first 10 lines
                    if line.startswith('文件: '):
                        filename = line.replace('文件: ', '').strip()
                        print(f"📄 Original filename in EEG: {filename}")
                        
                        # Extract subject ID and video number from the original filename
                        # Format: A00055837_video2_video2_extracted_AvgRef.csv
                        match = re.match(r'([A-Z0-9]+)_video(\d+)_.*', filename)
                        if match:
                            subject_id = match.group(1)
                            video_num = match.group(2)
                            return subject_id, video_num
                        
                        # Alternative format: A00053375_video1_extracted_AvgRef.csv
                        match = re.match(r'([A-Z0-9]+)_video(\d+)_.*', filename)
                        if match:
                            subject_id = match.group(1)
                            video_num = match.group(2)
                            return subject_id, video_num
                            
                return None, None
                
        except Exception as e:
            print(f"❌ Error reading EEG file {eeg_file}: {e}")
            return None, None
    
    def load_eeg_data(self, eeg_file):
        """Load EEG data from SPRiNT parameters log file"""
        try:
            channel_data = {}
            current_channel = None
            
            with open(eeg_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Detect channel start
                    if '=== 通道' in line and '参数提取开始' in line:
                        match = re.search(r'通道 (\d+)', line)
                        if match:
                            current_channel = match.group(1)
                            channel_data[current_channel] = {
                                'offset': [],
                                'exponent': []
                            }
                    
                    # Extract offset and exponent values
                    elif current_channel and '偏移(offset):' in line:
                        match = re.search(r'偏移\(offset\): ([-\d.]+)', line)
                        if match:
                            channel_data[current_channel]['offset'].append(float(match.group(1)))
                    
                    elif current_channel and '指数(exponent):' in line:
                        match = re.search(r'指数\(exponent\): ([-\d.]+)', line)
                        if match:
                            channel_data[current_channel]['exponent'].append(float(match.group(1)))
            
            if not channel_data:
                print(f"❌ No channel data found in {eeg_file}")
                return None
            
            print(f"✅ Loaded EEG data: {len(channel_data)} channels")
            return channel_data
            
        except Exception as e:
            print(f"❌ Error loading EEG data from {eeg_file}: {e}")
            return None
    
    def load_pupil_data(self, pupil_file):
        """Load pupil size data from log file"""
        try:
            with open(pupil_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter out comment lines and empty lines
            times = []
            pupil_sizes = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            time_val = float(parts[0])
                            pupil_val = float(parts[1])
                            times.append(time_val)
                            pupil_sizes.append(pupil_val)
                        except ValueError:
                            continue
            
            if len(times) == 0:
                print(f"❌ No valid data found in {pupil_file}")
                return None
            
            pupil_data = {
                'times': np.array(times),
                'pupil_sizes': np.array(pupil_sizes)
            }
            
            print(f"✅ Loaded pupil data: {len(times)} samples")
            return pupil_data
            
        except Exception as e:
            print(f"❌ Error loading pupil data from {pupil_file}: {e}")
            return None
    
    def align_signals(self, pupil_data, eeg_channel_data):
        """Align pupil and EEG signals in time (robust to NaNs and length mismatch)"""
        try:
            pupil_times = pupil_data['times']
            pupil_values = pupil_data['pupil_sizes']
            valid_pupil = np.isfinite(pupil_values)
            pupil_times_valid = pupil_times[valid_pupil]
            pupil_values_valid = pupil_values[valid_pupil]
            if len(pupil_times_valid) < 5:
                print("⚠️ Not enough valid pupil samples after NaN filtering")
                return None
            aligned_data = {
                'times': None,
                'pupil': None,
                'channels': {}
            }
            for channel, data in eeg_channel_data.items():
                if len(data['offset']) > 1 and len(data['exponent']) > 1:
                    ch_len = len(data['offset'])
                    end_time = min(int(pupil_times_valid.max()), ch_len - 1)
                    start_time = max(int(pupil_times_valid.min()), 0)
                    if end_time - start_time < 10:
                        continue
                    common_times = np.arange(start_time, end_time + 1)
                    pupil_aligned = np.interp(common_times, pupil_times_valid, pupil_values_valid)
                    channel_times = np.arange(ch_len)
                    valid_mask = (channel_times >= start_time) & (channel_times <= end_time)
                    if np.sum(valid_mask) < 2:
                        continue
                    valid_times = channel_times[valid_mask]
                    valid_offset = np.array(data['offset'])[valid_mask]
                    valid_exponent = np.array(data['exponent'])[valid_mask]
                    offset_aligned = np.interp(common_times, valid_times, valid_offset)
                    exponent_aligned = np.interp(common_times, valid_times, valid_exponent)
                    aligned_data['channels'][channel] = {
                        'offset': offset_aligned,
                        'exponent': exponent_aligned
                    }
                    aligned_data['times'] = common_times
                    aligned_data['pupil'] = pupil_aligned
            if not aligned_data['channels']:
                print("⚠️ No valid channels after alignment")
                return None
            return aligned_data
        except Exception as e:
            print(f"❌ Error aligning signals: {e}")
            return None
    
    def normalize_signal(self, signal_data):
        """Normalize signal using z-score"""
        signal_data = np.array(signal_data)
        if np.std(signal_data) == 0:
            return signal_data
        return (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    def calculate_coupling_metrics(self, pupil_signal, eeg_signal):
        """Calculate coupling metrics between pupil and EEG signals"""
        try:
            # Normalize signals
            pupil_norm = self.normalize_signal(pupil_signal)
            eeg_norm = self.normalize_signal(eeg_signal)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(pupil_norm) & np.isfinite(eeg_norm)
            if np.sum(valid_mask) < 10:  # Need minimum valid points
                return None
            
            pupil_clean = pupil_norm[valid_mask]
            eeg_clean = eeg_norm[valid_mask]
            
            # Calculate Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(pupil_clean, eeg_clean)
            
            # Calculate Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(pupil_clean, eeg_clean)
            
            # Calculate cross-correlation at zero lag
            cross_corr = np.corrcoef(pupil_clean, eeg_clean)[0, 1]
            
            # Calculate mutual information (simplified version)
            # Bin the data and calculate mutual information
            bins = 10
            pupil_binned = np.digitize(pupil_clean, np.linspace(pupil_clean.min(), pupil_clean.max(), bins))
            eeg_binned = np.digitize(eeg_clean, np.linspace(eeg_clean.min(), eeg_clean.max(), bins))
            
            # Calculate joint and marginal probabilities
            joint_hist, _, _ = np.histogram2d(pupil_binned, eeg_binned, bins=bins)
            joint_prob = joint_hist / np.sum(joint_hist)
            
            pupil_prob = np.sum(joint_prob, axis=1)
            eeg_prob = np.sum(joint_prob, axis=0)
            
            # Calculate mutual information
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if joint_prob[i, j] > 0 and pupil_prob[i] > 0 and eeg_prob[j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (pupil_prob[i] * eeg_prob[j]))
            
            results = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'cross_correlation': cross_corr,
                'mutual_information': mi,
                'n_samples': len(pupil_clean)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error calculating coupling metrics: {e}")
            return None
    
    def process_single_pair(self, data_pair):
        """Process a single EEG-pupil data pair"""
        subject_id = data_pair['subject_id']
        video_id = data_pair['video_id']
        
        print(f"\n🔄 Processing {subject_id}_video{video_id}...")
        
        # Load data
        eeg_data = self.load_eeg_data(data_pair['eeg_file'])
        pupil_data = self.load_pupil_data(data_pair['pupil_file'])
        
        if eeg_data is None or pupil_data is None:
            print(f"❌ Failed to load data for {subject_id}_video{video_id}")
            return None
        
        # Align signals
        aligned_data = self.align_signals(pupil_data, eeg_data)
        if aligned_data is None:
            print(f"❌ Failed to align signals for {subject_id}_video{video_id}")
            return None
        
        # Calculate coupling for each channel
        results = []
        
        for channel, channel_data in aligned_data['channels'].items():
            offset_coupling = self.calculate_coupling_metrics(aligned_data['pupil'], channel_data['offset'])
            exponent_coupling = self.calculate_coupling_metrics(aligned_data['pupil'], channel_data['exponent'])
            if offset_coupling is None and exponent_coupling is None:
                continue
            result = {
                'subject_id': subject_id,
                'video_id': video_id,
                'channel': channel,
                'offset_pearson_r': offset_coupling['pearson_r'] if offset_coupling else np.nan,
                'offset_pearson_p': offset_coupling['pearson_p'] if offset_coupling else np.nan,
                'offset_spearman_r': offset_coupling['spearman_r'] if offset_coupling else np.nan,
                'offset_spearman_p': offset_coupling['spearman_p'] if offset_coupling else np.nan,
                'offset_cross_corr': offset_coupling['cross_correlation'] if offset_coupling else np.nan,
                'offset_mutual_info': offset_coupling['mutual_information'] if offset_coupling else np.nan,
                'exponent_pearson_r': exponent_coupling['pearson_r'] if exponent_coupling else np.nan,
                'exponent_pearson_p': exponent_coupling['pearson_p'] if exponent_coupling else np.nan,
                'exponent_spearman_r': exponent_coupling['spearman_r'] if exponent_coupling else np.nan,
                'exponent_spearman_p': exponent_coupling['spearman_p'] if exponent_coupling else np.nan,
                'exponent_cross_corr': exponent_coupling['cross_correlation'] if exponent_coupling else np.nan,
                'exponent_mutual_info': exponent_coupling['mutual_information'] if exponent_coupling else np.nan,
                'n_samples': (offset_coupling or exponent_coupling)['n_samples']
            }
            results.append(result)
        
        print(f"✅ Processed {len(results)} channels for {subject_id}_video{video_id}")
        return results
    
    def run_batch_analysis(self):
        """Run batch analysis on all data pairs"""
        print("🚀 Starting batch PAC analysis...")
        
        # Find all data pairs
        data_pairs = self.find_data_pairs()
        
        if not data_pairs:
            print("❌ No data pairs found!")
            return
        
        # Process all pairs
        all_results = []
        successful_pairs = 0
        
        for i, data_pair in enumerate(data_pairs):
            print(f"\n📊 Processing pair {i+1}/{len(data_pairs)}")
            
            try:
                results = self.process_single_pair(data_pair)
                if results:
                    all_results.extend(results)
                    successful_pairs += 1
            except Exception as e:
                print(f"❌ Error processing {data_pair['subject_id']}_video{data_pair['video_id']}: {e}")
        
        # Save results
        if all_results:
            self.save_results(all_results)
            print(f"\n🎉 Analysis complete!")
            print(f"📊 Successfully processed: {successful_pairs}/{len(data_pairs)} pairs")
            print(f"📈 Total coupling measurements: {len(all_results)}")
        else:
            print("❌ No results to save!")
    
    def save_results(self, results):
        """Save results to CSV file"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            detailed_file = os.path.join(self.results_dir, f"pac_coupling_detailed_{timestamp}.csv")
            df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
            print(f"📁 Detailed results saved to: {detailed_file}")
            
            # Create summary by subject and video
            summary_data = []
            for subject_id in df['subject_id'].unique():
                for video_id in df['video_id'].unique():
                    subset = df[(df['subject_id'] == subject_id) & (df['video_id'] == video_id)]
                    if len(subset) > 0:
                        summary = {
                            'subject_id': subject_id,
                            'video_id': video_id,
                            'n_channels': len(subset),
                            'offset_pearson_mean': subset['offset_pearson_r'].mean(),
                            'offset_pearson_std': subset['offset_pearson_r'].std(),
                            'offset_spearman_mean': subset['offset_spearman_r'].mean(),
                            'offset_spearman_std': subset['offset_spearman_r'].std(),
                            'offset_mutual_info_mean': subset['offset_mutual_info'].mean(),
                            'offset_mutual_info_std': subset['offset_mutual_info'].std(),
                            'exponent_pearson_mean': subset['exponent_pearson_r'].mean(),
                            'exponent_pearson_std': subset['exponent_pearson_r'].std(),
                            'exponent_spearman_mean': subset['exponent_spearman_r'].mean(),
                            'exponent_spearman_std': subset['exponent_spearman_r'].std(),
                            'exponent_mutual_info_mean': subset['exponent_mutual_info'].mean(),
                            'exponent_mutual_info_std': subset['exponent_mutual_info'].std(),
                            'avg_n_samples': subset['n_samples'].mean()
                        }
                        summary_data.append(summary)
            
            # Save summary results
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.results_dir, f"pac_coupling_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"📁 Summary results saved to: {summary_file}")
            
            # Print some statistics
            print(f"\n📊 Analysis Statistics:")
            print(f"   - Total subjects: {df['subject_id'].nunique()}")
            print(f"   - Total videos: {len(summary_df)}")
            print(f"   - Average channels per video: {summary_df['n_channels'].mean():.1f}")
            print(f"   - Offset coupling (Pearson): {summary_df['offset_pearson_mean'].mean():.3f} ± {summary_df['offset_pearson_mean'].std():.3f}")
            print(f"   - Exponent coupling (Pearson): {summary_df['exponent_pearson_mean'].mean():.3f} ± {summary_df['exponent_pearson_mean'].std():.3f}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function to run the batch PAC analysis"""
    print("=" * 60)
    print("🧠 PAC Analysis - Batch Processing")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PACAnalysisBatch()
    
    # Run batch analysis
    analyzer.run_batch_analysis()
    
    print("\n" + "=" * 60)
    print("✅ Analysis completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
