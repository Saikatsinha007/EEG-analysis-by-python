# Complete app.py - Advanced Medical EEG Analysis App
import streamlit as st
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, fft
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')
import antropy as ant
import yasa
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from datetime import datetime
import tempfile
import os
import base64
import json
import textwrap

# Page configuration
st.set_page_config(
    page_title="NeuroVision EEG Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Modern Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #F1F5F9;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
    }
    .metric-card h4 {
        color: #64748B;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #64748B;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    /* Alerts */
    .alert-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid transparent;
        animation: slideIn 0.3s ease-out;
    }
    .critical-alert {
        background-color: #FEF2F2;
        border-left-color: #EF4444;
        color: #991B1B;
    }
    .warning-alert {
        background-color: #FFFBEB;
        border-left-color: #F59E0B;
        color: #92400E;
    }
    .info-alert {
        background-color: #EFF6FF;
        border-left-color: #3B82F6;
        color: #1E40AF;
    }
    .success-alert {
        background-color: #F0FDF4;
        border-left-color: #22C55E;
        color: #166534;
    }
    
    /* Welcome Screen */
    .hero-box {
        background: linear-gradient(135deg, #4F46E5 0%, #0EA5E9 100%);
        border-radius: 24px;
        padding: 3rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin: 2rem 0;
        text-align: left;
    }
    .hero-features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    .feature-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Components */
    .stButton button {
        background: #3B82F6;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
        transition: all 0.2s;
        width: 100%;
    }
    .stButton button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

class AdvancedEEGAnalyzer:
    def __init__(self, edf_path):
        """Initialize with advanced preprocessing"""
        try:
            self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            self.original_raw = self.raw.copy()
            self.sfreq = self.raw.info['sfreq']
            self.ch_names = self.raw.ch_names
            self.duration = self.raw.n_times / self.sfreq
            self.file_loaded = True
            self.artifacts_detected = {}
            self.seizure_events = []
            self.sleep_stages = []
            self.epoch_metrics = {}
            self.spike_wave_complexes = []
            
            # Standardize channel names if possible
            self.standardize_channel_names()
            
        except Exception as e:
            st.error(f"❌ Error loading EDF file: {str(e)}")
            self.file_loaded = False
    
    def standardize_channel_names(self):
        """Standardize channel names to 10-20 system"""
        if not self.raw:
            return

        new_names = {}
        for ch in self.raw.ch_names:
            # Clean name
            clean = ch.replace('EEG ', '').replace('POL ', '').strip()
            clean = clean.replace('-Ref', '').replace('-AVG', '').replace(' Ref', '')
            # Handle some specific common variations
            if clean.endswith('-0'): clean = clean[:-2]
            
            # Additional cleanup for common artifacts in EDF exports
            clean = clean.split('-')[0]  # Take first part of bipolar ref like Fp1-F7
            
            if clean != ch:
                new_names[ch] = clean
        
        if new_names:
            try:
                self.raw.rename_channels(new_names)
                self.ch_names = self.raw.ch_names
            except Exception as e:
                # print(f"Could not rename channels: {e}")
                self.ch_names = self.raw.ch_names
        else:
            self.ch_names = self.raw.ch_names
    
    def preprocess_for_analysis(self):
        """Comprehensive preprocessing pipeline"""
        if not self.file_loaded:
            return None
        
        try:
            # 1. Apply montage if possible
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                self.raw.set_montage(montage)
            except:
                pass
            
            # 2. Apply bandpass filter (0.5-70 Hz for clinical EEG)
            self.raw.filter(0.5, 70., fir_design='firwin', verbose=False)
            
            # 3. Notch filter for power line interference
            self.raw.notch_filter([50, 60], verbose=False)
            
            # 4. Re-reference to average (common in clinical EEG)
            self.raw.set_eeg_reference('average', verbose=False)
            
            # 5. Resample if too high sampling rate
            if self.sfreq > 500:
                self.raw.resample(250, verbose=False)
                self.sfreq = 250
            
            return self.raw
            
        except Exception as e:
            st.warning(f"⚠️ Preprocessing limited: {str(e)}")
            return self.raw
    
    def detect_seizure_patterns(self):
        """Advanced seizure detection using multiple features"""
        seizure_features = []
        
        # Make sure self.ch_names exists and is iterable
        if not hasattr(self, 'ch_names') or not self.ch_names:
            return seizure_features
        
        for ch in self.ch_names[:10]:  # Analyze first 10 channels
            data = self.get_channel_data(ch)
            
            # Check if data is valid
            if data is None or len(data) == 0:
                continue
                
            # Use only first 30 seconds
            max_samples = min(len(data), int(30 * self.sfreq))
            if max_samples < int(5 * self.sfreq):  # Need at least 5 seconds
                continue
                
            segment = data[:max_samples]
            
            # 1. Hjorth Parameters
            try:
                activity = np.var(segment)
                if activity == 0: continue
                
                diff1 = np.diff(segment)
                if np.std(diff1) == 0: continue
                mobility = np.std(diff1) / np.std(segment)
                
                diff2 = np.diff(diff1)
                if np.std(diff2) == 0: continue
                complexity = np.std(diff2) / np.std(diff1) / mobility
            except:
                complexity = 0
            
            # 2. Line Length Feature (common in seizure detection)
            line_length = np.sum(np.abs(np.diff(segment)))
            
            # 3. Energy
            energy = np.sum(segment ** 2)
            
            # 4. Statistical features
            try:
                skew = stats.skew(segment)
                kurt = stats.kurtosis(segment)
            except:
                skew, kurt = 0, 0
            
            # 5. Spectral features
            try:
                freqs, psd = signal.welch(segment, self.sfreq, nperseg=min(256, len(segment)))
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
                
                alpha_power = np.sum(psd[alpha_idx]) if np.any(alpha_idx) else 0
                beta_power = np.sum(psd[beta_idx]) if np.any(beta_idx) else 0
                alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
            except:
                alpha_beta_ratio = 0
            
            # 6. Spike detection
            # Make sure detect_spikes returns a list
            spikes = self.detect_spikes(segment, ch)
            if not isinstance(spikes, list):
                spikes = []  # Default to empty list
            
            # Seizure likelihood score
            spike_count = len(spikes) if isinstance(spikes, list) else 0
            
            seizure_score = (
                line_length / len(segment) * 100 +
                complexity * 10 +
                (alpha_beta_ratio > 2) * 50 +
                spike_count * 20
            )
            
            if seizure_score > 50 or spike_count > 5:  # Threshold
                seizure_features.append({
                    'channel': ch,
                    'score': seizure_score,
                    'time': 0,  # Starting time
                    'duration': 5,  # Estimated duration in seconds
                    'spike_count': spike_count,
                    'features': {
                        'line_length': line_length,
                        'complexity': complexity,
                        'alpha_beta_ratio': alpha_beta_ratio,
                        'spikes': spikes[:5] if len(spikes) > 0 else []
                    }
                })
        
        self.seizure_events = seizure_features
        return seizure_features
    
    def detect_spikes(self, data, channel):
        """Detect epileptiform spikes - returns list"""
        spikes = []
        
        if data is None or len(data) == 0:
            return spikes
        
        try:
            # Detect peaks
            peaks, properties = signal.find_peaks(
                data, 
                height=np.std(data) * 3 if np.std(data) > 0 else 0,
                distance=int(self.sfreq * 0.1) if hasattr(self, 'sfreq') else 10
            )
            
            for peak in peaks[:20]:  # Limit to first 20 spikes
                if peak > 20 and peak < len(data) - 20:
                    segment = data[peak-20:peak+20]
                    diff_segment = np.diff(segment)
                    
                    if np.max(np.abs(diff_segment)) > (np.std(data) * 5 if np.std(data) > 0 else 0):
                        spikes.append({
                            'position': peak / self.sfreq if hasattr(self, 'sfreq') else 0,
                            'amplitude': data[peak],
                            'channel': channel
                        })
        except Exception as e:
            print(f"Error in spike detection: {e}")
        
        return spikes  # Make sure this returns the list
    
    def detect_spike_wave_complexes(self):
        """Detect 3Hz spike-wave complexes (typical of absence seizures)"""
        complexes = []
        
        for ch in self.ch_names[:8]:  # Check first 8 channels
            data = self.get_channel_data(ch)[:int(10 * self.sfreq)]  # First 10 seconds
            
            if len(data) < self.sfreq * 3:
                continue
            
            # Bandpass filter for 2.5-3.5 Hz (spike-wave frequency)
            b, a = signal.butter(4, [2.5/(self.sfreq/2), 3.5/(self.sfreq/2)], btype='band')
            filtered = signal.filtfilt(b, a, data)
            
            # Find rhythmic activity
            autocorr = np.correlate(filtered, filtered, mode='same')
            autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
            
            # Look for peaks at ~3Hz period
            period_samples = int(self.sfreq / 3)  # Expected period for 3Hz
            search_range = range(period_samples-10, period_samples+10)
            
            for lag in search_range:
                if lag < len(autocorr) and autocorr[lag] > np.mean(autocorr)*2:
                    complexes.append({
                        'channel': ch,
                        'frequency': 3.0,
                        'strength': autocorr[lag],
                        'time': 0
                    })
                    break
        
        self.spike_wave_complexes = complexes
        return complexes
    
    def detect_sleep_patterns(self):
        """Automatic sleep stage scoring using YASA"""
        try:
            # Extract 30-second epochs (standard for sleep scoring)
            data = self.raw.get_data(picks=['Fz', 'Cz', 'Oz'])[:3, :int(300 * self.sfreq)]  # 5 minutes
            
            # Use YASA for sleep staging
            sls = yasa.SleepStaging(
                data, 
                sf=self.sfreq,
                eeg_name='Cz' if 'Cz' in self.raw.ch_names else self.raw.ch_names[0]
            )
            
            hypno = sls.predict()
            self.sleep_stages = hypno
            
            # Map YASA predictions to sleep stages
            stage_map = {
                0: 'Wake',
                1: 'N1',
                2: 'N2',
                3: 'N3',
                4: 'N3',  # N3 and N4 combined
                5: 'REM'
            }
            
            stages = [stage_map.get(h, 'Unknown') for h in hypno]
            
            # Calculate sleep metrics
            total_epochs = len(hypno)
            wake_epochs = sum(1 for h in hypno if h == 0)
            rem_epochs = sum(1 for h in hypno if h == 5)
            deep_sleep_epochs = sum(1 for h in hypno if h in [3, 4])
            
            return {
                'stages': stages,
                'hypnogram': hypno,
                'sleep_efficiency': np.mean(np.array(hypno) > 0) * 100,
                'rem_percentage': (rem_epochs / total_epochs) * 100 if total_epochs > 0 else 0,
                'deep_sleep_percentage': (deep_sleep_epochs / total_epochs) * 100 if total_epochs > 0 else 0,
                'sleep_latency': next((i for i, h in enumerate(hypno) if h > 0), None)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_sleep_spindles(self):
        """Detect sleep spindles (important for sleep staging)"""
        spindles = []
        
        if 'Cz' in self.ch_names:
            data = self.get_channel_data('Cz')[:int(60 * self.sfreq)]  # First minute
            
            # Filter in spindle range (11-16 Hz)
            b, a = signal.butter(4, [11/(self.sfreq/2), 16/(self.sfreq/2)], btype='band')
            filtered = signal.filtfilt(b, a, data)
            
            # Detect spindle-like activity
            envelope = np.abs(signal.hilbert(filtered))
            threshold = np.mean(envelope) + np.std(envelope)
            
            # Find spindle events
            above_threshold = envelope > threshold
            changes = np.diff(above_threshold.astype(int))
            start_indices = np.where(changes == 1)[0]
            end_indices = np.where(changes == -1)[0]
            
            for start, end in zip(start_indices[:10], end_indices[:10]):  # First 10
                duration = (end - start) / self.sfreq
                if 0.5 <= duration <= 2.0:  # Typical spindle duration
                    spindles.append({
                        'start_time': start / self.sfreq,
                        'duration': duration,
                        'amplitude': np.max(envelope[start:end]),
                        'channel': 'Cz'
                    })
        
        return spindles
    
    def detect_k_complexes(self):
        """Detect K-complexes (important for N2 sleep)"""
        k_complexes = []
        
        if 'Fz' in self.ch_names:
            data = self.get_channel_data('Fz')[:int(60 * self.sfreq)]  # First minute
            
            # Low pass filter to see slow waves
            b, a = signal.butter(4, 4/(self.sfreq/2), btype='low')
            filtered = signal.filtfilt(b, a, data)
            
            # Find negative peaks
            peaks, _ = signal.find_peaks(-filtered, 
                                        height=np.std(filtered)*2,
                                        distance=int(self.sfreq))  # 1 second minimum
            
            for peak in peaks[:10]:  # First 10
                # Check for characteristic shape
                if peak > 100 and peak < len(data) - 100:
                    window = filtered[peak-100:peak+100]
                    
                    # Should have sharp negative followed by positive
                    if (filtered[peak] < -np.std(filtered)*2 and 
                        np.max(window[100:150]) > np.std(filtered)):
                        k_complexes.append({
                            'time': peak / self.sfreq,
                            'amplitude': filtered[peak],
                            'channel': 'Fz'
                        })
        
        return k_complexes
    
    def detect_artifacts_comprehensive(self):
        """Comprehensive artifact detection"""
        artifacts = {
            'ocular': [],
            'muscle': [],
            'cardiac': [],
            'electrode_pop': [],
            'movement': [],
            'sweat': []
        }
        
        # Check if ch_names exists
        if not hasattr(self, 'ch_names') or not self.ch_names:
            return artifacts
        
        for ch in self.ch_names:
            data = self.get_channel_data(ch)
            
            # Skip if no data
            if data is None or len(data) == 0:
                continue
            
            # Make sure welch returns proper values
            try:
                freqs, psd = signal.welch(data, self.sfreq, nperseg=min(256, len(data)//4))
                if len(freqs) == 0: continue
                
                # 1. Detect high-frequency muscle artifacts (EMG)
                high_freq_idx = freqs > 30
                emg_power = np.sum(psd[high_freq_idx])
                
                if emg_power > np.mean(psd) * 10:
                    artifacts['muscle'].append({
                        'channel': ch,
                        'severity': 'High',
                        'confidence': min(emg_power / (np.mean(psd) + 1e-10) / 10, 1.0),
                        'power_ratio': emg_power / (np.sum(psd) + 1e-10)
                    })
                
                # 2. Detect ocular artifacts (slow, high amplitude)
                low_freq_idx = freqs < 4
                low_freq_power = np.sum(psd[low_freq_idx])
                total_power = np.sum(psd)
                
                if low_freq_power / (total_power + 1e-10) > 0.5:  # More than 50% power in low freq
                    artifacts['ocular'].append({
                        'channel': ch,
                        'severity': 'Medium',
                        'confidence': low_freq_power / (total_power + 1e-10),
                        'dominant_freq': freqs[np.argmax(psd[low_freq_idx])] if np.any(low_freq_idx) else 0
                    })
                
                # 3. Detect electrode pops (sharp transients)
                diff_signal = np.abs(np.diff(data))
                pop_threshold = np.median(data) + 5 * np.std(data)
                pop_indices = np.where(diff_signal > pop_threshold)[0]
                
                if len(pop_indices) > 0:
                    artifacts['electrode_pop'].append({
                        'channel': ch,
                        'count': len(pop_indices),
                        'locations': pop_indices / self.sfreq,  # Convert to seconds
                        'rate_per_minute': len(pop_indices) / (len(data) / self.sfreq / 60)
                    })
                
                # 4. Detect sweat/slow drift artifacts
                if len(data) > self.sfreq * 10:  # Need at least 10 seconds
                    # High pass filter at 0.1 Hz
                    b, a = signal.butter(2, 0.1/(self.sfreq/2), btype='high')
                    filtered = signal.filtfilt(b, a, data)
                    
                    # Compare variance
                    original_var = np.var(data)
                    filtered_var = np.var(filtered)
                    
                    if original_var > (filtered_var + 1e-10) * 10:
                        artifacts['sweat'].append({
                            'channel': ch,
                            'severity': 'Low',
                            'drift_ratio': original_var / (filtered_var + 1e-10)
                        })
                
            except Exception as e:
                # print(f"Error processing channel {ch}: {e}")
                continue
        
        self.artifacts_detected = artifacts
        return artifacts
    
    def compute_advanced_metrics(self):
        """Compute comprehensive clinical metrics"""
        metrics = {}
        
        # 1. Global metrics
        all_data = self.raw.get_data()
        metrics['global'] = {
            'mean_amplitude': np.mean(np.abs(all_data)),
            'amplitude_range': np.max(all_data) - np.min(all_data),
            'signal_variance': np.var(all_data),
            'dynamic_range': 20 * np.log10(np.max(np.abs(all_data)) / np.min(np.abs(all_data[all_data != 0]))),
            'snr_estimate': np.mean(all_data) / np.std(all_data),
            'total_power': np.sum(all_data**2)
        }
        
        # 2. Per-channel metrics
        channel_metrics = {}
        for ch in self.ch_names[:15]:  # First 15 channels for performance
            data = self.get_channel_data(ch)
            
            # Time-domain features
            hjorth_activity = np.var(data)
            hjorth_mobility = np.std(np.diff(data)) / np.std(data)
            hjorth_complexity = np.std(np.diff(np.diff(data))) / np.std(np.diff(data)) / hjorth_mobility
            
            channel_metrics[ch] = {
                'rms': np.sqrt(np.mean(data**2)),
                'peak_to_peak': np.max(data) - np.min(data),
                'zero_crossings': len(np.where(np.diff(np.sign(data)))[0]),
                'hjorth_activity': hjorth_activity,
                'hjorth_mobility': hjorth_mobility,
                'hjorth_complexity': hjorth_complexity,
                'hjorth_mobility': hjorth_mobility,
                'hjorth_complexity': hjorth_complexity
            }
            
            try:
                # Add entropy features safely
                channel_metrics[ch].update({
                    'spectral_entropy': ant.spectral_entropy(data, sf=self.sfreq, method='welch'),
                    'sample_entropy': ant.sample_entropy(data, metric='chebyshev') if len(data) > 100 else 0,
                    'hurst_exponent': ant.hurst_exponent(data) if len(data) > 100 else 0,
                    'detrended_fluctuation': ant.detrended_fluctuation(data) if len(data) > 100 else 0,
                })
            except Exception as e:
                # Fallback for entropy calculation errors
                channel_metrics[ch].update({
                    'spectral_entropy': 0, 'sample_entropy': 0, 
                    'hurst_exponent': 0.5, 'detrended_fluctuation': 0
                })

            channel_metrics[ch]['nonlinearity'] = hjorth_complexity / hjorth_mobility if hjorth_mobility > 0 else 0

        
        metrics['channels'] = channel_metrics
        
        # 3. Asymmetry indices (important for clinical diagnosis)
        if len(self.ch_names) >= 10:
            left_channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 'P7']
            right_channels = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']
            
            left_power, right_power = 0, 0
            left_count, right_count = 0, 0
            
            for lch, rch in zip(left_channels, right_channels):
                if lch in self.ch_names and rch in self.ch_names:
                    ldata = self.get_channel_data(lch)
                    rdata = self.get_channel_data(rch)
                    
                    lfreqs, lpsd = signal.welch(ldata, self.sfreq, nperseg=256)
                    rfreqs, rpsd = signal.welch(rdata, self.sfreq, nperseg=256)
                    
                    left_power += np.sum(lpsd)
                    right_power += np.sum(rpsd)
                    left_count += 1
                    right_count += 1
            
            if left_count > 0 and right_count > 0:
                metrics['asymmetry'] = {
                    'left_power_avg': left_power / left_count,
                    'right_power_avg': right_power / right_count,
                    'asymmetry_index': (right_power - left_power) / (right_power + left_power) * 100,
                    'symmetry_ratio': min(left_power, right_power) / max(left_power, right_power)
                }
        
        # 4. Reactivity metrics
        metrics['reactivity'] = self.compute_reactivity()
        
        # 5. Connectivity metrics (simplified)
        metrics['connectivity'] = self.compute_connectivity()
        
        self.epoch_metrics = metrics
        return metrics
    
    def compute_reactivity(self):
        """Compute alpha reactivity (eye opening/closing effect)"""
        if 'O1' in self.ch_names and 'O2' in self.ch_names:
            occipital_data = self.raw.get_data(picks=['O1', 'O2']).mean(axis=0)
            
            # Split into 5-second epochs
            segment_len = int(5 * self.sfreq)
            n_segments = len(occipital_data) // segment_len
            
            alpha_powers = []
            for i in range(min(10, n_segments)):
                segment = occipital_data[i*segment_len:(i+1)*segment_len]
                freqs, psd = signal.welch(segment, self.sfreq, nperseg=256)
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                if np.any(alpha_idx):
                    alpha_powers.append(np.trapz(psd[alpha_idx], freqs[alpha_idx]))
            
            if len(alpha_powers) > 1:
                reactivity_score = np.std(alpha_powers) / np.mean(alpha_powers)
                
                return {
                    'alpha_variability': reactivity_score,
                    'has_reactivity': reactivity_score > 0.3,
                    'alpha_power_mean': np.mean(alpha_powers),
                    'alpha_power_std': np.std(alpha_powers),
                    'reactivity_class': 'High' if reactivity_score > 0.5 else 
                                       'Moderate' if reactivity_score > 0.2 else 'Low'
                }
        
        return {'error': 'Occipital channels not available or insufficient data'}
    
    def compute_connectivity(self):
        """Compute simple connectivity between channels"""
        connectivity = {}
        
        if len(self.ch_names) >= 4:
            # Compute correlation matrix for first 8 channels
            channels = self.ch_names[:8]
            data_matrix = self.raw.get_data(picks=channels)
            
            # Correlation matrix
            corr_matrix = np.corrcoef(data_matrix)
            
            # Average connectivity
            avg_connectivity = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            
            connectivity = {
                'avg_correlation': avg_connectivity,
                'connectivity_class': 'High' if avg_connectivity > 0.7 else 
                                     'Moderate' if avg_connectivity > 0.4 else 'Low',
                'correlation_matrix': corr_matrix.tolist() if corr_matrix.shape[0] <= 8 else []
            }
        
        return connectivity
    
    def generate_clinical_report(self):
        """Generate comprehensive clinical report"""
        report = {
            'patient_info': {
                'recording_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file_duration': f"{self.duration:.1f} seconds",
                'sampling_rate': f"{self.sfreq} Hz",
                'channels_available': len(self.ch_names),
                'channels_list': self.ch_names[:20]
            },
            'quality_assessment': {
                'signal_quality': 'Good' if np.mean(self.raw.get_data()) > 10 else 'Poor',
                'artifact_level': sum(len(v) for v in self.artifacts_detected.values()),
                'impedance_estimate': 'Unknown',
                'recording_continuity': 'Continuous' if self.duration > 60 else 'Short',
                'noise_level': 'Low' if self.epoch_metrics.get('global', {}).get('snr_estimate', 0) > 2 else 'High'
            },
            'findings': {
                'seizure_activity': len(self.seizure_events) > 0,
                'seizure_count': len(self.seizure_events),
                'spike_wave_complexes': len(self.spike_wave_complexes),
                'sleep_patterns': len(self.sleep_stages) > 0,
                'artifact_present': any(len(v) > 0 for v in self.artifacts_detected.values()),
                'asymmetry_present': 'asymmetry' in self.epoch_metrics and 
                                    abs(self.epoch_metrics['asymmetry'].get('asymmetry_index', 0)) > 20,
                'abnormal_connectivity': self.epoch_metrics.get('connectivity', {}).get('avg_correlation', 0) > 0.8
            },
            'interpretation': [],
            'recommendations': []
        }
        
        # Generate interpretation
        if report['findings']['seizure_activity']:
            report['interpretation'].append(
                "⚡ **Epileptiform Activity**: Potential seizure patterns detected."
            )
        
        if report['findings']['spike_wave_complexes'] > 0:
            report['interpretation'].append(
                f"🌀 **Spike-Wave Complexes**: {len(self.spike_wave_complexes)} possible absence seizure patterns."
            )
        
        if report['findings']['asymmetry_present']:
            asym_index = self.epoch_metrics['asymmetry']['asymmetry_index']
            side = 'right' if asym_index > 0 else 'left'
            report['interpretation'].append(
                f"⚖️ **Hemispheric Asymmetry**: {abs(asym_index):.1f}% {side} hemisphere dominance."
            )
        
        if report['findings']['sleep_patterns']:
            sleep_info = self.detect_sleep_patterns()
            if 'sleep_efficiency' in sleep_info:
                report['interpretation'].append(
                    f"😴 **Sleep Architecture**: {sleep_info['sleep_efficiency']:.1f}% sleep efficiency."
                )
        
        # Generate recommendations
        if report['findings']['seizure_activity']:
            report['recommendations'].append(
                "⚠️ **Urgent**: Review by epileptologist recommended. Consider video-EEG monitoring."
            )
        
        if report['quality_assessment']['artifact_level'] > 20:
            report['recommendations'].append(
                "🔄 **Technical**: High artifact levels detected. Consider reapplication of electrodes."
            )
        
        if report['quality_assessment']['noise_level'] == 'High':
            report['recommendations'].append(
                "🔊 **Technical**: High noise level. Check electrode impedances (<5kΩ recommended)."
            )
        
        if report['findings']['asymmetry_present'] and abs(asym_index) > 30:
            report['recommendations'].append(
                "🧠 **Clinical**: Significant asymmetry detected. Correlate with neuroimaging findings."
            )
        
        if not report['findings']['seizure_activity'] and not report['findings']['asymmetry_present']:
            report['recommendations'].append(
                "✅ **Normal**: No significant abnormalities detected in this recording segment."
            )
        
        return report
    
    def get_channel_data(self, ch_name):
        """Extract data for specific channel - DIRECT from MNE"""
        if not hasattr(self, 'raw') or self.raw is None:
            return np.array([])
        
        # Try to find the channel in raw object
        if ch_name in self.raw.ch_names:
            try:
                # Use string indexing which is safer with MNE
                data = self.raw.get_data(picks=ch_name)
                return data.flatten()
            except:
                return np.array([])
        
        return np.array([])

# Main Application Functions
def main():
    # Main header
    st.markdown('<h1 class="main-header">⚡ NeuroVision EEG Pro - Advanced Clinical EEG Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### *Comprehensive EEG Analysis System for Neurologists*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Data Management")
        
        uploaded_file = st.file_uploader("Upload EDF File", type=['edf', 'EDF'])
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name
            
            try:
                with st.spinner("Loading and preprocessing EEG data..."):
                    analyzer = AdvancedEEGAnalyzer(file_path)
                    if analyzer.file_loaded:
                        analyzer.preprocess_for_analysis()
                        st.session_state.analyzer = analyzer
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                        
                        # Show basic info
                        st.markdown("---")
                        st.markdown("### 📊 Recording Info")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Duration", f"{analyzer.duration:.1f}s")
                        with col2:
                            st.metric("Channels", len(analyzer.ch_names))
                        
                        # Auto-run initial analyses
                        analyzer.detect_seizure_patterns()
                        analyzer.detect_artifacts_comprehensive()
                        analyzer.compute_advanced_metrics()
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.markdown("---")
        st.markdown("## 🔧 Analysis Settings")
        
        if st.session_state.analyzer:
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["Comprehensive", "Seizure Focus", "Sleep Study", "Artifact Review", "Quick Overview"]
            )
            
            with st.expander("🔌 Channel Info (Debug)"):
                st.write(f"Loaded: {len(st.session_state.analyzer.ch_names)} chans")
                st.write(st.session_state.analyzer.ch_names)
            
            st.markdown("---")
            st.markdown("## 📋 Navigation")
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Dashboard", use_container_width=True):
                    st.session_state.current_tab = "Dashboard"
            with col2:
                if st.button("📈 EEG Viewer", use_container_width=True):
                    st.session_state.current_tab = "EEG Viewer"
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔬 Analysis", use_container_width=True):
                    st.session_state.current_tab = "Analysis"
            with col4:
                if st.button("📋 Report", use_container_width=True):
                    st.session_state.current_tab = "Report"
    
    # Main content area
    if st.session_state.analyzer:
        analyzer = st.session_state.analyzer
        
        # Tab navigation
        tabs = st.tabs(["📊 Dashboard", "📈 EEG Viewer", "🔬 Advanced Analysis", "📋 Clinical Report"])
        
        with tabs[0]:  # Dashboard
            create_clinical_dashboard_tab(analyzer)
        
        with tabs[1]:  # EEG Viewer
            create_eeg_viewer_tab(analyzer)
        
        with tabs[2]:  # Advanced Analysis
            create_advanced_analysis_tab(analyzer)
        

        
        with tabs[3]:  # Clinical Report
            create_clinical_report_tab(analyzer)
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(textwrap.dedent("""
            <div style='text-align: center; padding: 3rem 1rem;'>
              <h2 style='font-size: 2rem; font-weight: 800; color: #1E293B;'>Welcome to NeuroVision EEG Pro</h2>
              <p style='font-size: 1.2rem; color: #64748B; margin-bottom: 2rem;'>
                Next-Generation Clinical Analysis Suite
              </p>
              
              <div class="hero-box">
                <h3 style='margin: 0; font-size: 1.5rem;'>🚀 Powerful Capabilities</h3>
                <div class="hero-features">
                  <div class="feature-item">⚡ Auto-Seizure Detection</div>
                  <div class="feature-item">😴 Sleep Staging</div>
                  <div class="feature-item">🔄 Artifact Removal</div>
                  <div class="feature-item">🧠 Connectivity Maps</div>
                  <div class="feature-item">📊 Clinical Reports</div>

                </div>
              </div>
              
              <p style='margin-top: 2rem; color: #64748B;'>
                <strong>Upload an EDF file</strong> to begin your analysis session
              </p>
            </div>
            """), unsafe_allow_html=True)

def create_clinical_dashboard_tab(analyzer):
    """Create the clinical dashboard tab"""
    
    st.markdown('<h2 class="section-header">📊 Clinical Dashboard</h2>', unsafe_allow_html=True)
    
    # Alert System
    st.markdown('<h3 class="section-header">⚠️ Clinical Alerts</h3>', unsafe_allow_html=True)
    
    # Run detection algorithms
    seizures = analyzer.detect_seizure_patterns()
    artifacts = analyzer.detect_artifacts_comprehensive()
    metrics = analyzer.compute_advanced_metrics()
    spike_waves = analyzer.detect_spike_wave_complexes()
    
    # Critical alerts
    alerts = []
    
    if len(seizures) > 0:
        alert_level = "critical" if len(seizures) > 5 else "warning"
        unique_channels = sorted(list(set([s['channel'] for s in seizures[:3]])))
        channels_str = ', '.join(unique_channels)
        avg_confidence = np.mean([s['score'] for s in seizures])
        
        seizure_alert = textwrap.dedent(f"""
        <div class="alert-box {alert_level}-alert">
          <h3>⚡ SEIZURE ACTIVITY DETECTED</h3>
          <p><strong>Events:</strong> {len(seizures)} potential seizure patterns</p>
          <p><strong>Channels:</strong> {channels_str}</p>
          <p><strong>Average confidence:</strong> {avg_confidence:.1f}%</p>
          <p><strong>Action:</strong> Immediate review recommended</p>
        </div>
        """)
        alerts.append(seizure_alert)
    
    if len(spike_waves) > 0:
        spike_alert = textwrap.dedent(f"""
        <div class="alert-box warning-alert">
          <h3>🌀 SPIKE-WAVE COMPLEXES</h3>
          <p><strong>Count:</strong> {len(spike_waves)} possible absence seizure patterns</p>
          <p><strong>Frequency:</strong> ~3Hz (typical for absence seizures)</p>
          <p><strong>Action:</strong> Review for generalized epilepsy patterns</p>
        </div>
        """)
        alerts.append(spike_alert)
    
    total_artifacts = sum(len(v) for v in artifacts.values())
    if total_artifacts > 20:
        primary_types = [k for k, v in artifacts.items() if len(v) > 5]
        types_str = ', '.join(primary_types)
        
        artifact_alert = textwrap.dedent(f"""
        <div class="alert-box info-alert">
          <h3>🔄 EXCESSIVE ARTIFACTS</h3>
          <p><strong>Total artifacts:</strong> {total_artifacts}</p>
          <p><strong>Primary types:</strong> {types_str}</p>
          <p><strong>Action:</strong> Consider data cleaning or re-recording</p>
        </div>
        """)
        alerts.append(artifact_alert)
    
    # Display alerts or success message
    if alerts:
        for alert in alerts:
            st.markdown(alert, unsafe_allow_html=True)
    else:
        st.markdown(textwrap.dedent("""
        <div class="alert-box success-alert">
          <h3>✅ NO CRITICAL FINDINGS</h3>
          <p>Initial screening shows no epileptiform activity or excessive artifacts.</p>
          <p>Proceed with detailed analysis for comprehensive assessment.</p>
        </div>
        """), unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown('<h3 class="section-header">📈 Key Metrics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        signal_quality = "✅ Good" if metrics['global']['snr_estimate'] > 2 else "⚠️ Check"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card">
            <h4>Signal Quality</h4>
            <h2>{signal_quality}</h2>
            <p>SNR: {metrics['global']['snr_estimate']:.2f}</p>
        </div>
        """), unsafe_allow_html=True)
    
    with col2:
        asym_index = metrics.get('asymmetry', {}).get('asymmetry_index', 0)
        asym_status = "✅ Balanced" if abs(asym_index) < 15 else "⚠️ Asymmetric"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card">
            <h4>Hemispheric Balance</h4>
            <h2>{asym_status}</h2>
            <p>Index: {asym_index:.1f}%</p>
        </div>
        """), unsafe_allow_html=True)
    
    with col3:
        total_power = metrics['global']['total_power']
        power_status = "Normal" if 1e6 < total_power < 1e9 else "Check"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card">
            <h4>Total Power</h4>
            <h2>{total_power:.2e}</h2>
            <p>Status: {power_status}</p>
        </div>
        """), unsafe_allow_html=True)
    
    with col4:
        artifact_count = total_artifacts
        artifact_status = "✅ Low" if artifact_count < 10 else "⚠️ High"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card">
            <h4>Artifact Count</h4>
            <h2>{artifact_count}</h2>
            <p>Status: {artifact_status}</p>
        </div>
        """), unsafe_allow_html=True)
    
    # Quick Insights
    st.markdown('<h3 class="section-header">🔍 Quick Insights</h3>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Dominant Frequency
        if 'Cz' in analyzer.ch_names:
            data = analyzer.get_channel_data('Cz')
            freqs, psd = signal.welch(data, analyzer.sfreq, nperseg=1024)
            dominant_freq = freqs[np.argmax(psd)]
            
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
              <h4>🎯 Dominant Frequency</h4>
              <h2>{dominant_freq:.1f} Hz</h2>
              <p>Central channel (Cz)</p>
              <p><small>{"Alpha" if 8 <= dominant_freq <= 13 else 
                        "Beta" if 13 < dominant_freq <= 30 else 
                        "Theta" if 4 <= dominant_freq < 8 else 
                        "Delta" if dominant_freq < 4 else "Gamma"} rhythm dominant</small></p>
            </div>
            """), unsafe_allow_html=True)
    
    with insights_col2:
        # Reactivity Score
        reactivity = metrics.get('reactivity', {})
        if 'reactivity_class' in reactivity:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
              <h4>👁️ Alpha Reactivity</h4>
              <h2>{reactivity['reactivity_class']}</h2>
              <p>Occipital response</p>
              <p><small>Variability: {reactivity.get('alpha_variability', 0):.3f}</small></p>
            </div>
            """), unsafe_allow_html=True)
    
    # Sleep Analysis (if applicable)
    if analyzer.duration > 300:  # More than 5 minutes
        st.markdown('<h3 class="section-header">😴 Sleep Analysis</h3>', unsafe_allow_html=True)
        
        sleep_info = analyzer.detect_sleep_patterns()
        if 'error' not in sleep_info:
            sleep_col1, sleep_col2, sleep_col3 = st.columns(3)
            
            with sleep_col1:
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card">
                    <h4>Sleep Efficiency</h4>
                    <h2>{sleep_info['sleep_efficiency']:.1f}%</h2>
                    <p>Quality Score</p>
                </div>
                """), unsafe_allow_html=True)
            
            with sleep_col2:
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card">
                    <h4>REM Sleep</h4>
                    <h2>{sleep_info['rem_percentage']:.1f}%</h2>
                    <p>of total sleep</p>
                </div>
                """), unsafe_allow_html=True)
            
            with sleep_col3:
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card">
                    <h4>Deep Sleep</h4>
                    <h2>{sleep_info['deep_sleep_percentage']:.1f}%</h2>
                    <p>N3 Stage</p>
                </div>
                """), unsafe_allow_html=True)
            
            # Sleep hypnogram
            if len(sleep_info['hypnogram']) > 0:
                fig_hypno = go.Figure()
                fig_hypno.add_trace(go.Scatter(
                    y=sleep_info['hypnogram'],
                    mode='lines+markers',
                    line=dict(width=3, color='blue'),
                    marker=dict(size=8)
                ))
                fig_hypno.update_layout(
                    title='Sleep Hypnogram',
                    yaxis=dict(
                        title='Sleep Stage',
                        tickvals=[0, 1, 2, 3, 5],
                        ticktext=['Wake', 'N1', 'N2', 'N3', 'REM']
                    ),
                    xaxis_title='Epochs (30s each)',
                    height=300
                )
                st.plotly_chart(fig_hypno, use_container_width=True)

def create_eeg_viewer_tab(analyzer):
    """Create the EEG viewer tab"""
    
    st.markdown('<h2 class="section-header">📈 Advanced EEG Viewer</h2>', unsafe_allow_html=True)
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        montage_type = st.selectbox(
            "Montage Type",
            ["Referential", "Bipolar (Longitudinal)", "Bipolar (Transverse)", "Average Reference"]
        )
    
    with col2:
        scale_factor = st.slider("Scale (μV)", 10, 200, 50)
        
    with col3:
        height_per_channel = st.slider("Vertical Spacing", 50, 300, 150)
        
    col4, col5 = st.columns(2)
    with col4:
        time_window = st.slider("Time Window (seconds)", 2, 60, 10)
    with col5:
        start_time = st.slider("Start Time (seconds)", 0, int(analyzer.duration - time_window), 0)
    
    # Channel selection
    st.markdown("### Channel Selection")
    
    # Group channels by region
    frontal = [ch for ch in analyzer.ch_names if ch.startswith('F')]
    central = [ch for ch in analyzer.ch_names if ch.startswith('C')]
    parietal = [ch for ch in analyzer.ch_names if ch.startswith('P')]
    occipital = [ch for ch in analyzer.ch_names if ch.startswith('O')]
    temporal = [ch for ch in analyzer.ch_names if ch.startswith('T')]
    
    selected_channels = []
    
    regions = ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]
    region_channels = [frontal, central, parietal, occipital, temporal]
    
    cols = st.columns(len(regions))
    cols = st.columns(len(regions))
    for idx, (region, channels) in enumerate(zip(regions, region_channels)):
        with cols[idx]:
            if channels:
                if st.checkbox(region, value=True, key=f"region_{region}"):
                    selected_channels.extend(channels[:3])  # Limit to 3 per region
    
    if not selected_channels:
        st.warning("⚠️ Auto-grouping failed. Please select channels manually below.")
        selected_channels = st.multiselect(
            "Select Channels to View", 
            analyzer.ch_names,
            default=analyzer.ch_names[:min(5, len(analyzer.ch_names))]
        )
        
    if not selected_channels:
         st.error("No channels selected or available.")
         return
    
    # Create EEG display
    fig = make_subplots(
        rows=len(selected_channels), 
        cols=1,
        subplot_titles=selected_channels,
        vertical_spacing=0.02,
        shared_xaxes=True
    )
    
    start_idx = int(start_time * analyzer.sfreq)
    end_idx = int((start_time + time_window) * analyzer.sfreq)
    times = analyzer.raw.times[start_idx:end_idx]
    
    # Color mapping for regions
    region_colors = {
        'F': '#3B82F6',  # Blue
        'C': '#10B981',  # Green
        'P': '#F59E0B',  # Orange
        'O': '#8B5CF6',  # Purple
        'T': '#EF4444'   # Red
    }
    
    for i, ch in enumerate(selected_channels, 1):
        if ch in analyzer.ch_names:
            data = analyzer.get_channel_data(ch)
            segment = data[start_idx:end_idx]
            
            # Get color based on region
            color = region_colors.get(ch[0], '#666666')
            
            # Normalize for display
            if np.std(segment) > 0:
                segment = (segment - np.mean(segment)) / np.std(segment) * scale_factor
            
            # Add offset for stacking
            y_offset = (len(selected_channels) - i) * scale_factor * 3
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=segment + y_offset,
                    mode='lines',
                    name=ch,
                    line=dict(width=1.5, color=color),
                    hoverinfo='x+y+name'
                ),
                row=i, col=1
            )
            
            # Add zero line
            fig.add_hline(y=y_offset, line_dash="dot", line_width=1, line_color="gray", row=i, col=1)
    
    fig.update_layout(
        height=height_per_channel * len(selected_channels),
        showlegend=False,
        title=f"EEG Display - {montage_type} Montage",
        xaxis_title="Time (seconds)",
        yaxis=dict(showticklabels=False, showgrid=True, zeroline=True),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional controls
    st.markdown("### 📊 Signal Processing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Apply Filter", use_container_width=True):
            st.info("Filter applied: 0.5-70 Hz bandpass")
    
    with col2:
        if st.button("Remove Artifacts", use_container_width=True):
            st.info("ICA-based artifact removal would be applied here")
    
    with col3:
        if st.button("Export View", use_container_width=True):
            st.info("View exported as PNG")

def create_advanced_analysis_tab(analyzer):
    """Create advanced analysis tab"""
    
    st.markdown('<h2 class="section-header">🔬 Advanced Analysis</h2>', unsafe_allow_html=True)
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Spectral Analysis", "Time-Frequency Analysis", "Connectivity Analysis", 
         "Nonlinear Dynamics", "Artifact Characterization", "Seizure Detection"]
    )
    
    if analysis_type == "Spectral Analysis":
        create_spectral_analysis(analyzer)
    
    elif analysis_type == "Time-Frequency Analysis":
        create_time_frequency_analysis(analyzer)
    
    elif analysis_type == "Connectivity Analysis":
        create_connectivity_analysis(analyzer)
    
    elif analysis_type == "Nonlinear Dynamics":
        create_nonlinear_analysis(analyzer)
    
    elif analysis_type == "Artifact Characterization":
        create_artifact_analysis(analyzer)
    
    elif analysis_type == "Seizure Detection":
        create_seizure_detection_analysis(analyzer)

def create_spectral_analysis(analyzer):
    """Spectral analysis section"""
    
    st.markdown("### 📊 Power Spectral Density Analysis")
    
    # Channel selection
    selected_channel = st.selectbox(
        "Select Channel for Analysis",
        analyzer.ch_names[:15] if analyzer.ch_names else []
    )
    
    if selected_channel:
        data = analyzer.get_channel_data(selected_channel)
        
        # Calculate PSD
        freqs, psd = signal.welch(data, analyzer.sfreq, nperseg=1024)
        
        # Define frequency bands
        bands = {
            'Delta (0.5-4 Hz)': (0.5, 4, '#FF6B6B'),
            'Theta (4-8 Hz)': (4, 8, '#4ECDC4'),
            'Alpha (8-13 Hz)': (8, 13, '#45B7D1'),
            'Beta (13-30 Hz)': (13, 30, '#96CEB4'),
            'Gamma (30-50 Hz)': (30, 50, '#FECA57')
        }
        
        # Calculate band powers
        band_powers = {}
        for band_name, (low, high, color) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                band_powers[band_name] = np.trapz(psd[idx], freqs[idx])
            else:
                band_powers[band_name] = 0
        
        # Create two columns for plots
        col1, col2 = st.columns(2)
        
        with col1:
            # PSD plot
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(
                x=freqs, 
                y=10*np.log10(psd + 1e-10),
                mode='lines',
                name='PSD',
                line=dict(color='blue', width=2)
            ))
            
            # Add band shading
            for band_name, (low, high, color) in bands.items():
                fig_psd.add_vrect(
                    x0=low, x1=high,
                    fillcolor=color, opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=band_name.split()[0],
                    annotation_position="top left"
                )
            
            fig_psd.update_layout(
                title=f'Power Spectral Density - {selected_channel}',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Power (dB/Hz)',
                height=400
            )
            st.plotly_chart(fig_psd, use_container_width=True)
        
        with col2:
            # Band power distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(band_powers.keys()),
                values=list(band_powers.values()),
                hole=0.4,
                marker_colors=[bands[k][2] for k in band_powers.keys()],
                textinfo='label+percent',
                textposition='inside'
            )])
            
            fig_pie.update_layout(
                title='Frequency Band Distribution',
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Band power metrics
        st.markdown("### 📈 Band Power Metrics")
        
        cols = st.columns(5)
        for idx, (band_name, power) in enumerate(band_powers.items()):
            with cols[idx]:
                band_short = band_name.split()[0]
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{band_short}</h4>
                    <h3>{power:.2e}</h3>
                    <p>{band_name}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Alpha/Beta ratio and other metrics
        alpha_power = band_powers.get('Alpha (8-13 Hz)', 1e-10)
        beta_power = band_powers.get('Beta (13-30 Hz)', 1e-10)
        theta_power = band_powers.get('Theta (4-8 Hz)', 1e-10)
        delta_power = band_powers.get('Delta (0.5-4 Hz)', 1e-10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha_beta_ratio = alpha_power / beta_power
            st.metric("Alpha/Beta Ratio", f"{alpha_beta_ratio:.2f}",
                     delta="High" if alpha_beta_ratio > 2 else "Normal" if alpha_beta_ratio > 1 else "Low")
        
        with col2:
            theta_beta_ratio = theta_power / beta_power
            st.metric("Theta/Beta Ratio", f"{theta_beta_ratio:.2f}",
                     delta="High" if theta_beta_ratio > 2 else "Normal")
        
        with col3:
            delta_theta_ratio = delta_power / theta_power
            st.metric("Delta/Theta Ratio", f"{delta_theta_ratio:.2f}",
                     delta="High" if delta_theta_ratio > 1.5 else "Normal")

def create_time_frequency_analysis(analyzer):
    """Time-frequency analysis section"""
    
    st.markdown("### ⏱️ Time-Frequency Analysis")
    
    selected_channel = st.selectbox(
        "Select Channel",
        analyzer.ch_names[:8] if analyzer.ch_names else []
    )
    
    if selected_channel:
        # Get data (first 30 seconds for performance)
        data = analyzer.get_channel_data(selected_channel)[:int(30 * analyzer.sfreq)]
        
        # Create spectrogram
        f, t, Sxx = spectrogram(data, analyzer.sfreq, nperseg=256, noverlap=128)
        
        fig = go.Figure(data=go.Heatmap(
            z=10*np.log10(Sxx + 1e-10),
            x=t,
            y=f,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))
        
        fig.update_layout(
            title=f'Spectrogram - {selected_channel}',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional TFA metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Mean frequency over time
            mean_freq = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
            fig_mean = go.Figure()
            fig_mean.add_trace(go.Scatter(
                x=t, y=mean_freq,
                mode='lines',
                line=dict(color='red', width=2)
            ))
            fig_mean.update_layout(
                title='Mean Frequency Over Time',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                height=300
            )
            st.plotly_chart(fig_mean, use_container_width=True)
        
        with col2:
            # Band power evolution
            alpha_idx = np.logical_and(f >= 8, f <= 13)
            beta_idx = np.logical_and(f >= 13, f <= 30)
            
            alpha_power = np.sum(Sxx[alpha_idx, :], axis=0)
            beta_power = np.sum(Sxx[beta_idx, :], axis=0)
            
            fig_bands = go.Figure()
            fig_bands.add_trace(go.Scatter(
                x=t, y=alpha_power, name='Alpha', line=dict(color='blue')
            ))
            fig_bands.add_trace(go.Scatter(
                x=t, y=beta_power, name='Beta', line=dict(color='green')
            ))
            fig_bands.update_layout(
                title='Alpha/Beta Power Evolution',
                xaxis_title='Time (s)',
                yaxis_title='Power',
                height=300
            )
            st.plotly_chart(fig_bands, use_container_width=True)

def create_connectivity_analysis(analyzer):
    """Connectivity analysis section"""
    
    st.markdown("### 🧠 Brain Connectivity Analysis")
    
    # Select channels for connectivity
    available_channels = analyzer.ch_names[:8]
    selected_channels = st.multiselect(
        "Select Channels for Connectivity Analysis",
        available_channels,
        default=available_channels[:4]
    )
    
    if len(selected_channels) >= 2:
        # Get data for selected channels
        data_matrix = []
        for ch in selected_channels:
            data = analyzer.get_channel_data(ch)[:int(10 * analyzer.sfreq)]
            data_matrix.append(data)
        
        data_matrix = np.array(data_matrix)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_matrix)
        
        # Plot correlation matrix
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=selected_channels,
            y=selected_channels,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title='Inter-channel Correlation Matrix',
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Connectivity metrics
        st.markdown("### 📊 Connectivity Metrics")
        
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        correlations = corr_matrix[triu_indices]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_correlation = np.mean(np.abs(correlations))
            st.metric("Average Connectivity", f"{avg_correlation:.3f}",
                     delta="Strong" if avg_correlation > 0.7 else 
                           "Moderate" if avg_correlation > 0.4 else "Weak")
        
        with col2:
            max_correlation = np.max(np.abs(correlations))
            st.metric("Maximum Connectivity", f"{max_correlation:.3f}")
        
        with col3:
            connectivity_entropy = -np.sum(correlations * np.log(correlations + 1e-10))
            st.metric("Connectivity Complexity", f"{connectivity_entropy:.3f}")
        
        # Network visualization
        st.markdown("### 🌐 Network Visualization")
        
        # Create network graph
        edge_trace = []
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            text=selected_channels,
            textposition="top center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Simple circular layout
        n_nodes = len(selected_channels)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 1
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        node_trace.x = x
        node_trace.y = y
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if abs(corr_matrix[i, j]) > 0.3:  # Threshold
                    edge_x.extend([x[i], x[j], None])
                    edge_y.extend([y[i], y[j], None])
                    edge_weights.append(abs(corr_matrix[i, j]))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            mode='lines',
            hoverinfo='none'
        )
        
        fig_network = go.Figure(data=[edge_trace, node_trace])
        fig_network.update_layout(
            title='Brain Connectivity Network',
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig_network, use_container_width=True)

def create_nonlinear_analysis(analyzer):
    """Nonlinear dynamics analysis"""
    
    st.markdown("### 🌀 Nonlinear Dynamics Analysis")
    
    selected_channel = st.selectbox(
        "Select Channel",
        analyzer.ch_names[:10] if analyzer.ch_names else []
    )
    
    if selected_channel:
        data = analyzer.get_channel_data(selected_channel)[:int(10 * analyzer.sfreq)]
        
        # Calculate nonlinear features
        try:
            sampen = ant.sample_entropy(data)
            df = ant.detrended_fluctuation(data)
            hurst = ant.hurst_exponent(data)
            perm_entropy = ant.perm_entropy(data, order=3, delay=1, normalize=True)
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sample Entropy", f"{sampen:.3f}",
                         delta="High complexity" if sampen > 1.5 else "Low complexity")
            
            with col2:
                st.metric("DFA α", f"{df:.3f}",
                         delta="Long-range" if df > 0.8 else "Short-range")
            
            with col3:
                st.metric("Hurst Exponent", f"{hurst:.3f}",
                         delta="Persistent" if hurst > 0.5 else "Anti-persistent")
            
            with col4:
                st.metric("Permutation Entropy", f"{perm_entropy:.3f}")
            
            # Phase space reconstruction
            st.markdown("#### Phase Space Analysis")
            
            # Create delay embedding
            delay = int(analyzer.sfreq * 0.01)  # 10ms delay
            embedded = nk.complexity_embedding(data, delay=delay, dimension=3)
            
            if len(embedded) > 100:
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=embedded[:500, 0],
                    y=embedded[:500, 1],
                    z=embedded[:500, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=np.arange(500),
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    line=dict(width=1, color='darkblue')
                )])
                
                fig_3d.update_layout(
                    title='Phase Space Reconstruction',
                    scene=dict(
                        xaxis_title='x(t)',
                        yaxis_title='x(t+τ)',
                        zaxis_title='x(t+2τ)'
                    ),
                    height=500
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Recurrence plot
            st.markdown("#### Recurrence Analysis")
            
            # Simplified recurrence plot
            rp_size = 200
            if len(data) > rp_size:
                segment = data[:rp_size]
                recurrence_matrix = np.zeros((rp_size, rp_size))
                
                for i in range(rp_size):
                    for j in range(rp_size):
                        if np.abs(segment[i] - segment[j]) < np.std(segment) * 0.5:
                            recurrence_matrix[i, j] = 1
                
                fig_rp = go.Figure(data=go.Heatmap(
                    z=recurrence_matrix,
                    colorscale='Greys',
                    showscale=False
                ))
                
                fig_rp.update_layout(
                    title='Recurrence Plot',
                    xaxis_title='Time',
                    yaxis_title='Time',
                    height=400
                )
                
                st.plotly_chart(fig_rp, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Nonlinear analysis limited: {str(e)}")

def create_artifact_analysis(analyzer):
    """Artifact characterization analysis"""
    
    st.markdown("### 🔍 Artifact Characterization")
    
    artifacts = analyzer.detect_artifacts_comprehensive()
    
    # Display artifact summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ocular = len(artifacts['ocular'])
        st.metric("Ocular Artifacts", ocular)
    
    with col2:
        muscle = len(artifacts['muscle'])
        st.metric("Muscle Artifacts", muscle)
    
    with col3:
        electrode = len(artifacts['electrode_pop'])
        st.metric("Electrode Pops", electrode)
    
    with col4:
        movement = len(artifacts['movement'])
        st.metric("Movement Artifacts", movement)
    
    # Detailed artifact analysis
    st.markdown("#### 📊 Artifact Distribution by Channel")
    
    # Create artifact heatmap
    artifact_channels = {}
    for artifact_type, artifact_list in artifacts.items():
        for artifact in artifact_list:
            ch = artifact['channel']
            if ch not in artifact_channels:
                artifact_channels[ch] = {'total': 0}
            artifact_channels[ch]['total'] += 1
            artifact_channels[ch][artifact_type] = artifact_channels[ch].get(artifact_type, 0) + 1
    
    if artifact_channels:
        channels = list(artifact_channels.keys())[:15]
        artifact_types = ['ocular', 'muscle', 'electrode_pop', 'movement']
        
        data_matrix = []
        for ch in channels:
            row = [artifact_channels[ch].get(at, 0) for at in artifact_types]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix).T
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=channels,
            y=['Ocular', 'Muscle', 'Electrode Pop', 'Movement'],
            colorscale='YlOrRd',
            colorbar=dict(title="Count")
        ))
        
        fig_heatmap.update_layout(
            title='Artifact Distribution by Channel',
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Artifact timeline
    st.markdown("#### ⏱️ Artifact Timeline")
    
    # Select a channel to view artifact timeline
    if analyzer.ch_names:
        timeline_channel = st.selectbox(
            "Select Channel for Timeline View",
            analyzer.ch_names[:10]
        )
        
        if timeline_channel:
            data = analyzer.get_channel_data(timeline_channel)[:int(30 * analyzer.sfreq)]
            
            # Detect artifacts in this channel
            artifact_indices = []
            artifact_types = []
            
            # Simple artifact detection for visualization
            threshold = np.std(data) * 3
            peaks, _ = signal.find_peaks(np.abs(data), height=threshold)
            
            # Classify peaks as artifacts
            for peak in peaks[:50]:  # Limit for performance
                if peak < len(data):
                    # Check frequency content
                    if peak > 10 and peak < len(data) - 10:
                        segment = data[peak-10:peak+10]
                        freqs, psd = signal.welch(segment, analyzer.sfreq, nperseg=20)
                        
                        if np.any(freqs > 30) and psd[np.argmax(freqs > 30)] > np.mean(psd) * 2:
                            artifact_types.append('Muscle')
                        else:
                            artifact_types.append('Other')
                        
                        artifact_indices.append(peak / analyzer.sfreq)
            
            # Create timeline plot
            fig_timeline = go.Figure()
            
            # Add EEG signal
            times = np.arange(len(data)) / analyzer.sfreq
            fig_timeline.add_trace(go.Scatter(
                x=times, y=data,
                mode='lines',
                name='EEG Signal',
                line=dict(color='blue', width=1)
            ))
            
            # Add artifact markers
            if artifact_indices:
                for idx, art_type in zip(artifact_indices, artifact_types):
                    fig_timeline.add_vline(
                        x=idx,
                        line_dash="dash",
                        line_color="red" if art_type == 'Muscle' else "orange",
                        opacity=0.5
                    )
            
            fig_timeline.update_layout(
                title=f'Artifact Timeline - {timeline_channel}',
                xaxis_title='Time (s)',
                yaxis_title='Amplitude (μV)',
                height=400
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Artifact removal suggestions
    st.markdown("#### 🛠️ Artifact Removal Recommendations")
    
    recommendations = []
    
    if len(artifacts['ocular']) > 5:
        recommendations.append("""
        **👁️ Ocular Artifacts Detected:**
        - Apply ICA for eye blink removal
        - Use frontal bipolar montage
        - Consider EOG channel subtraction
        """)
    
    if len(artifacts['muscle']) > 5:
        recommendations.append("""
        **💪 Muscle Artifacts Detected:**
        - Apply 30-50 Hz band-stop filter
        - Use EMG channels for reference
        - Consider muscle relaxation period
        """)
    
    if len(artifacts['electrode_pop']) > 5:
        recommendations.append("""
        **⚡ Electrode Pops Detected:**
        - Check electrode connections
        - Reapply conductive gel
        - Use interpolation for bad channels
        """)
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ Minimal artifacts detected. Signal quality is good.")

def create_seizure_detection_analysis(analyzer):
    """Advanced seizure detection analysis"""
    
    st.markdown("### ⚡ Advanced Seizure Detection")
    
    # Run seizure detection
    seizures = analyzer.detect_seizure_patterns()
    spike_waves = analyzer.detect_spike_wave_complexes()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Seizure Events", len(seizures))
    
    with col2:
        st.metric("Spike-Wave Complexes", len(spike_waves))
    
    with col3:
        total_epileptiform = len(seizures) + len(spike_waves)
        st.metric("Total Epileptiform", total_epileptiform)
    
    # Seizure event details
    if seizures:
        st.markdown("#### 📋 Detected Seizure Events")
        
        # Create table of seizure events
        seizure_data = []
        for seizure in seizures[:10]:  # Show first 10
            seizure_data.append({
                'Channel': seizure['channel'],
                'Confidence': f"{seizure['score']:.1f}%",
                'Duration': f"{seizure['duration']}s",
                'Spikes': seizure.get('spike_count', 0),
                'Features': ', '.join([k for k, v in seizure['features'].items() 
                                      if isinstance(v, (int, float)) and v > 0])
            })
        
        if seizure_data:
            df_seizures = pd.DataFrame(seizure_data)
            st.dataframe(df_seizures, use_container_width=True)
        
        # Seizure channel distribution
        st.markdown("#### 📊 Seizure Distribution by Channel")
        
        channel_counts = {}
        for seizure in seizures:
            ch = seizure['channel']
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
        
        if channel_counts:
            channels = list(channel_counts.keys())
            counts = list(channel_counts.values())
            
            fig_bar = go.Figure(data=[go.Bar(
                x=channels, y=counts,
                marker_color='red'
            )])
            
            fig_bar.update_layout(
                title='Seizure Events by Channel',
                xaxis_title='Channel',
                yaxis_title='Number of Events',
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Spike-wave complex analysis
    if spike_waves:
        st.markdown("#### 🌀 Spike-Wave Complex Analysis")
        
        spike_data = []
        for complex in spike_waves[:10]:
            spike_data.append({
                'Channel': complex['channel'],
                'Frequency': f"{complex['frequency']:.1f} Hz",
                'Strength': f"{complex['strength']:.2e}",
                'Time': f"{complex['time']:.1f}s"
            })
        
        if spike_data:
            df_spikes = pd.DataFrame(spike_data)
            st.dataframe(df_spikes, use_container_width=True)
        
        # Clinical interpretation
        st.markdown("#### 🧠 Clinical Interpretation")
        
        if len(spike_waves) > 3:
            st.warning("""
            **Potential Absence Seizures Detected**
            
            **Clinical Significance:**
            - Typical 3Hz spike-wave pattern
            - Suggests generalized epilepsy
            - Common in childhood absence epilepsy
            
            **Recommended Actions:**
            1. Clinical correlation with patient history
            2. Consider hyperventilation test
            3. Evaluate for other seizure types
            4. Neurological consultation
            """)
    
    # Seizure evolution analysis
    st.markdown("#### 📈 Seizure Evolution Analysis")
    
    if analyzer.ch_names:
        seizure_channel = st.selectbox(
            "Select Channel for Seizure Evolution",
            analyzer.ch_names[:8]
        )
        
        if seizure_channel:
            # Analyze 60 seconds of data
            data = analyzer.get_channel_data(seizure_channel)[:int(60 * analyzer.sfreq)]
            
            # Calculate moving window features
            window_size = int(5 * analyzer.sfreq)  # 5-second windows
            step_size = int(analyzer.sfreq)  # 1-second steps
            
            features = []
            times = []
            
            for i in range(0, len(data) - window_size, step_size):
                window = data[i:i+window_size]
                
                # Calculate features
                line_length = np.sum(np.abs(np.diff(window)))
                energy = np.sum(window ** 2)
                variance = np.var(window)
                
                # Spectral features
                freqs, psd = signal.welch(window, analyzer.sfreq, nperseg=256)
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
                
                alpha_power = np.sum(psd[alpha_idx]) if np.any(alpha_idx) else 0
                beta_power = np.sum(psd[beta_idx]) if np.any(beta_idx) else 0
                
                features.append({
                    'line_length': line_length,
                    'energy': energy,
                    'variance': variance,
                    'alpha_beta_ratio': alpha_power / (beta_power + 1e-10) if beta_power > 0 else 0
                })
                times.append(i / analyzer.sfreq)
            
            if features and times:
                # Plot feature evolution
                fig_features = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Line Length', 'Energy', 'Variance', 'Alpha/Beta Ratio']
                )
                
                # Line length
                fig_features.add_trace(
                    go.Scatter(x=times, y=[f['line_length'] for f in features], mode='lines'),
                    row=1, col=1
                )
                
                # Energy
                fig_features.add_trace(
                    go.Scatter(x=times, y=[f['energy'] for f in features], mode='lines'),
                    row=1, col=2
                )
                
                # Variance
                fig_features.add_trace(
                    go.Scatter(x=times, y=[f['variance'] for f in features], mode='lines'),
                    row=2, col=1
                )
                
                # Alpha/Beta ratio
                fig_features.add_trace(
                    go.Scatter(x=times, y=[f['alpha_beta_ratio'] for f in features], mode='lines'),
                    row=2, col=2
                )
                
                fig_features.update_layout(
                    title=f'Seizure Feature Evolution - {seizure_channel}',
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig_features, use_container_width=True)
    
    if not seizures and not spike_waves:
        st.success("""
        ✅ **No epileptiform activity detected**
        
        **Interpretation:**
        - No seizure patterns identified in this recording
        - Normal EEG background activity
        - Continue monitoring as clinically indicated
        """)

def create_clinical_report_tab(analyzer):
    """Create clinical report tab"""
    
    st.markdown('<h2 class="section-header">📋 Comprehensive Clinical Report</h2>', unsafe_allow_html=True)
    
    # Generate comprehensive report
    report = analyzer.generate_clinical_report()
    
    # Report header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  padding: 2rem; border-radius: 10px; color: white; margin: 1rem 0;'>
            <h2>NeuroVision EEG Pro</h2>
            <h3>Clinical EEG Analysis Report</h3>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Patient and Recording Information
    st.markdown("### 📄 Patient & Recording Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        **Recording Details:**
        - Duration: {report['patient_info']['file_duration']}
        - Sampling Rate: {report['patient_info']['sampling_rate']}
        - Channels: {report['patient_info']['channels_available']}
        """)
    
    with info_col2:
        st.markdown(f"""
        **Analysis Details:**
        - Analysis Date: {report['patient_info']['recording_date']}
        - Software Version: NeuroVision EEG Pro 2.0
        - Analysis Mode: Comprehensive
        """)
    
    # Quality Assessment
    st.markdown("### 🔧 Quality Assessment")
    
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        sq_color = "green" if report['quality_assessment']['signal_quality'] == 'Good' else "red"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card" style="padding: 1rem;">
            <h4>Signal Quality</h4>
            <h2 style="color: {sq_color}">{report['quality_assessment']['signal_quality']}</h2>
        </div>
        """), unsafe_allow_html=True)
    
    with quality_cols[1]:
        al_color = "green" if report['quality_assessment']['artifact_level'] < 10 else "orange" if report['quality_assessment']['artifact_level'] < 20 else "red"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card" style="padding: 1rem;">
            <h4>Artifact Level</h4>
            <h2 style="color: {al_color}">{report['quality_assessment']['artifact_level']}</h2>
        </div>
        """), unsafe_allow_html=True)
    
    with quality_cols[2]:
        rc_color = "green" if report['quality_assessment']['recording_continuity'] == 'Continuous' else "orange"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card" style="padding: 1rem;">
            <h4>Continuity</h4>
            <h2 style="color: {rc_color}">{report['quality_assessment']['recording_continuity']}</h2>
        </div>
        """), unsafe_allow_html=True)
    
    with quality_cols[3]:
        nl_color = "green" if report['quality_assessment']['noise_level'] == 'Low' else "red"
        st.markdown(textwrap.dedent(f"""
        <div class="metric-card" style="padding: 1rem;">
            <h4>Noise Level</h4>
            <h2 style="color: {nl_color}">{report['quality_assessment']['noise_level']}</h2>
        </div>
        """), unsafe_allow_html=True)
    
    # Key Findings
    st.markdown("### 🔍 Key Findings")
    
    findings_col1, findings_col2 = st.columns(2)
    
    with findings_col1:
        findings_list = []
        
        if report['findings']['seizure_activity']:
            findings_list.append(f"⚡ **Epileptiform Activity:** {report['findings']['seizure_count']} events detected")
        
        if report['findings']['spike_wave_complexes'] > 0:
            findings_list.append(f"🌀 **Spike-Wave Complexes:** {report['findings']['spike_wave_complexes']} detected")
        
        if report['findings']['asymmetry_present']:
            findings_list.append("⚖️ **Significant Hemispheric Asymmetry**")
        
        if report['findings']['sleep_patterns']:
            findings_list.append("😴 **Sleep Architecture Identified**")
        
        if findings_list:
            for finding in findings_list:
                st.markdown(f"- {finding}")
        else:
            st.markdown("- ✅ **Normal EEG Patterns**")
    
    with findings_col2:
        # Additional metrics
        metrics = analyzer.compute_advanced_metrics()
        
        if 'global' in metrics:
            st.markdown(f"""
            **Signal Metrics:**
            - Mean Amplitude: {metrics['global']['mean_amplitude']:.2f} μV
            - Dynamic Range: {metrics['global']['dynamic_range']:.1f} dB
            - SNR Estimate: {metrics['global']['snr_estimate']:.2f}
            """)
        
        if 'asymmetry' in metrics:
            asym = metrics['asymmetry']
            st.markdown(f"""
            **Asymmetry Metrics:**
            - Asymmetry Index: {asym['asymmetry_index']:.1f}%
            - Symmetry Ratio: {asym.get('symmetry_ratio', 0):.2f}
            """)
    
    # Clinical Interpretation
    st.markdown("### 🧠 Clinical Interpretation")
    
    if report['interpretation']:
        for interpretation in report['interpretation']:
            st.markdown(interpretation)
    else:
        st.markdown("""
        **Normal EEG Findings:**
        - Background activity within normal limits
        - No epileptiform discharges detected
        - Symmetrical hemispheric activity
        - Appropriate reactivity to state changes
        """)
    
    # Recommendations
    st.markdown("### 💡 Clinical Recommendations")
    
    if report['recommendations']:
        for i, recommendation in enumerate(report['recommendations'], 1):
            st.markdown(f"{i}. {recommendation}")
    else:
        st.markdown("""
        1. ✅ **Normal Study:** No further action required based on EEG findings
        2. 📋 **Clinical Correlation:** Always correlate with clinical presentation
        3. 🔄 **Follow-up:** Routine follow-up as per clinical indication
        """)
    
    # Export options
    st.markdown("### 📤 Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.info("PDF report generation would be implemented here")
    
    with col2:
        if st.button("📊 Export Data", use_container_width=True):
            # Create export data
            export_data = {
                'report': report,
                'metrics': analyzer.epoch_metrics,
                'artifacts': analyzer.artifacts_detected,
                'seizures': analyzer.seizure_events,
                'recording_info': {
                    'duration': analyzer.duration,
                    'sampling_rate': analyzer.sfreq,
                    'channels': analyzer.ch_names
                }
            }
            
            # Convert to JSON for download
            json_str = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="eeg_analysis_report.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🖼️ Save Visualizations", use_container_width=True):
            st.info("Visualization export would be implemented here")
    
    # Detailed Metrics Table
    st.markdown("### 📊 Detailed Metrics")
    
    if 'channels' in analyzer.epoch_metrics:
        channel_metrics = analyzer.epoch_metrics['channels']
        
        # Create DataFrame for display
        metrics_list = []
        for channel, metrics_dict in list(channel_metrics.items())[:10]:  # First 10 channels
            row = {'Channel': channel}
            row.update({k: f"{v:.3e}" if isinstance(v, float) else str(v) 
                       for k, v in metrics_dict.items()})
            metrics_list.append(row)
        
        if metrics_list:
            df_metrics = pd.DataFrame(metrics_list)
            st.dataframe(df_metrics, use_container_width=True)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.code(traceback.format_exc())