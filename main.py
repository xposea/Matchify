import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, Optional, List, Tuple
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from functools import wraps
import random
from os import getenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=5, initial_delay=1):
    """Decorator for handling rate limiting with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            retries = 0

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except spotipy.SpotifyException as e:
                    if e.http_status == 429:
                        retry_after = int(e.headers.get('Retry-After', delay))
                        sleep_time = retry_after + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit. Waiting {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        delay *= 2
                        retries += 1
                    else:
                        raise
                except Exception as e:
                    raise

            raise Exception("Max retries exceeded")

        return wrapper

    return decorator


class SpotifyPlaybackManager:
    def __init__(self):
        """Initialize with proper authentication scope and rate limiting."""
        scope = " ".join([
            "streaming",
            "user-read-playback-state",
            "user-modify-playback-state",
            "user-top-read",
            "user-read-currently-playing",
            "user-read-recently-played"
        ])
        try:
            oauth_object = SpotifyOAuth(
                client_id=getenv('id'),
                client_secret=getenv('secret'),
                redirect_uri="http://google.com/callback/",
                scope=scope)
            self.player_token = oauth_object.get_access_token(as_dict=False)
            self.sp = spotipy.Spotify(auth=self.player_token)
            self.SEGMENT_DURATION = 15.0
            self.MIN_SEGMENTS = 5
            self.feature_weights = {
                'timbre': 0.25,
                'pitch': 0.25,
                'loudness': 0.15,
                'tempo': 0.15,
                'key_compatibility': 0.1,
                'rhythm': 0.1
            }
            self.request_count = 0
            self.last_request_time = time.time()
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise

    @retry_with_backoff()
    def _make_spotify_request(self, method, *args, **kwargs):
        """Make a rate-limited request to Spotify API."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)

        self.last_request_time = time.time()
        return method(*args, **kwargs)

    async def analyze_transition(self, track1_id: str, track2_id: str) -> Optional[Dict]:
        """Analyze transition between tracks with rate limiting."""
        try:
            analysis1 = self._make_spotify_request(self.sp.audio_analysis, track1_id)
            await asyncio.sleep(0.1)

            analysis2 = self._make_spotify_request(self.sp.audio_analysis, track2_id)
            await asyncio.sleep(0.1)

            features1 = self._make_spotify_request(self.sp.audio_features, [track1_id])[0]
            await asyncio.sleep(0.1)

            features2 = self._make_spotify_request(self.sp.audio_features, [track2_id])[0]
            await asyncio.sleep(0.1)

            track1 = self._make_spotify_request(self.sp.track, track1_id)
            await asyncio.sleep(0.1)

            track2 = self._make_spotify_request(self.sp.track, track2_id)

            duration1 = track1['duration_ms'] / 1000

            end_segments = self.get_time_bounded_segments(
                analysis1,
                duration1 - self.SEGMENT_DURATION,
                duration1
            )

            start_segments = self.get_time_bounded_segments(
                analysis2,
                0,
                self.SEGMENT_DURATION
            )

            if not end_segments or not start_segments:
                logger.warning("Insufficient segments for analysis")
                return None

            end_gradients = self.calculate_feature_gradients(end_segments)
            start_gradients = self.calculate_feature_gradients(start_segments)

            if not end_gradients or not start_gradients:
                logger.warning("Could not calculate gradients")
                return None

            compatibility = self.calculate_gradient_compatibility(end_gradients, start_gradients)

            tempo_diff = abs(features1['tempo'] - features2['tempo']) / max(features1['tempo'], features2['tempo'])
            key_score = self.calculate_key_compatibility(
                features1['key'], features1['mode'],
                features2['key'], features2['mode']
            )
            rhythm_score = self.analyze_rhythm_compatibility(end_segments, start_segments)

            final_scores = {
                'timbre': compatibility['timbre'],
                'pitch': compatibility['pitch'],
                'loudness': compatibility['loudness'],
                'tempo': tempo_diff,
                'key_compatibility': 1 - key_score,
                'rhythm': 1 - rhythm_score
            }

            weighted_score = sum(
                score * self.feature_weights[feature]
                for feature, score in final_scores.items()
            )

            final_scores['weighted_score'] = weighted_score
            final_scores['transition_data'] = {
                'end_segments': end_segments,
                'start_segments': start_segments,
                'track1_name': track1['name'],
                'track2_name': track2['name'],
                'track1_features': features1,
                'track2_features': features2
            }

            logger.info(f"Analyzed transition from {track1['name']} to {track2['name']}")
            return final_scores

        except Exception as e:
            logger.error(f"Error analyzing transition: {str(e)}")
            return None

    async def find_best_transitions(self, track_id: str, num_candidates: int = 40) -> Tuple[Dict, Dict]:
        """Find best transitions with rate limiting."""
        try:
            track_features = self._make_spotify_request(self.sp.audio_features, [track_id])[0]
            await asyncio.sleep(0.1)

            track_info = self._make_spotify_request(self.sp.track, track_id)
            await asyncio.sleep(0.1)

            recommendations = self._make_spotify_request(
                self.sp.recommendations,
                seed_tracks=[track_id],
                limit=num_candidates,
                target_tempo=track_features['tempo'],
                target_key=track_features['key'],
                target_mode=track_features['mode'],
                min_energy=max(0, track_features['energy'] - 0.2),
                max_energy=min(1, track_features['energy'] + 0.2),
                min_danceability=max(0, track_features['danceability'] - 0.2),
                max_danceability=min(1, track_features['danceability'] + 0.2)
            )

            best_intro = {'id': None, 'score': float('inf'), 'details': None}
            best_outro = {'id': None, 'score': float('inf'), 'details': None}

            logger.info(f"Analyzing {num_candidates} potential transitions...")

            for i, track in enumerate(recommendations['tracks']):
                candidate_id = track['id']
                if candidate_id == track_id:
                    continue

                logger.info(f"Progress: {i + 1}/{len(recommendations['tracks'])}")

                if i > 0:
                    await asyncio.sleep(0.2)

                intro_score = await self.analyze_transition(candidate_id, track_id)
                await asyncio.sleep(0.1)
                outro_score = await self.analyze_transition(track_id, candidate_id)

                if intro_score:
                    if intro_score['weighted_score'] < best_intro['score']:
                        best_intro = {
                            'id': candidate_id,
                            'score': intro_score['weighted_score'],
                            'details': intro_score,
                            'name': track['name'],
                            'artist': track['artists'][0]['name']
                        }

                if outro_score:
                    if outro_score['weighted_score'] < best_outro['score']:
                        best_outro = {
                            'id': candidate_id,
                            'score': outro_score['weighted_score'],
                            'details': outro_score,
                            'name': track['name'],
                            'artist': track['artists'][0]['name']
                        }

            return best_intro, best_outro

        except Exception as e:
            logger.error(f"Error finding transitions: {e}")
            return None, None

    @retry_with_backoff()
    def search_tracks(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for tracks with rate limiting."""
        try:
            results = self._make_spotify_request(self.sp.search, q=query, type='track', limit=limit)
            tracks = []

            for idx, track in enumerate(results['tracks']['items'], 1):
                if idx > 1:
                    time.sleep(0.1)

                audio_features = self._make_spotify_request(self.sp.audio_features, [track['id']])[0]

                tracks.append({
                    'index': idx,
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'duration': track['duration_ms'] / 1000,
                    'tempo': audio_features['tempo'],
                    'key': audio_features['key'],
                    'mode': audio_features['mode'],
                    'time_signature': audio_features['time_signature']
                })

            logger.info(f"Found {len(tracks)} tracks matching query: {query}")
            return tracks

        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []

    def calculate_key_compatibility(self, key1: int, mode1: int, key2: int, mode2: int) -> float:
        """Calculate musical key compatibility score."""
        circle_of_fifths = {
            0: [0, 7, 5],  # C
            1: [1, 8, 6],  # C#/Db
            2: [2, 9, 7],  # D
            3: [3, 10, 8],  # D#/Eb
            4: [4, 11, 9],  # E
            5: [5, 0, 10],  # F
            6: [6, 1, 11],  # F#/Gb
            7: [7, 2, 0],  # G
            8: [8, 3, 1],  # G#/Ab
            9: [9, 4, 2],  # A
            10: [10, 5, 3],  # A#/Bb
            11: [11, 6, 4]  # B
        }

        if key1 == key2 and mode1 == mode2:
            return 1.0
        elif key2 in circle_of_fifths[key1]:
            return 0.8
        else:
            return 0.4

    def get_time_bounded_segments(self, analysis: Dict, start_time: float, end_time: float) -> List:
        """Get segments within a specific time window."""
        segments = []
        current_time = 0

        try:
            for segment in analysis['segments']:
                if current_time >= start_time and current_time < end_time:
                    segment['start_time'] = current_time
                    segments.append(segment)
                elif current_time >= end_time:
                    break
                current_time += segment['duration']

            if len(segments) < self.MIN_SEGMENTS:
                logger.warning(f"Insufficient segments found: {len(segments)}")
                return []

            return segments
        except Exception as e:
            logger.error(f"Error in segment analysis: {e}")
            return []

    def calculate_feature_gradients(self, segments: List) -> Optional[Dict]:
        """Calculate feature gradients with improved error handling."""
        if not segments or len(segments) < self.MIN_SEGMENTS:
            return None

        try:
            gradients = {
                'timbre': [],
                'pitches': [],
                'loudness': []
            }

            for i in range(len(segments) - 1):
                dt = segments[i + 1]['start_time'] - segments[i]['start_time']
                if dt < 0.001:
                    continue

                timbre_grad = [(t2 - t1) / dt for t1, t2 in
                               zip(segments[i]['timbre'], segments[i + 1]['timbre'])]
                gradients['timbre'].append([g / max(1, abs(g)) for g in timbre_grad])

                pitch_grad = [(p2 - p1) / dt for p1, p2 in
                              zip(segments[i]['pitches'], segments[i + 1]['pitches'])]
                gradients['pitches'].append([g / max(1, abs(g)) for g in pitch_grad])

                loudness_grad = (segments[i + 1]['loudness_max'] - segments[i]['loudness_max']) / dt
                gradients['loudness'].append(loudness_grad / max(1, abs(loudness_grad)))

            return gradients
        except Exception as e:
            logger.error(f"Error calculating gradients: {e}")
            return None

    def calculate_gradient_compatibility(self, grad1: Dict, grad2: Dict) -> Dict:
        """Calculate gradient compatibility with improved normalization."""
        if not grad1 or not grad2:
            return {'timbre': 1.0, 'pitch': 1.0, 'loudness': 1.0}

        try:
            scores = {}

            window_size = min(3, len(grad1['timbre']), len(grad2['timbre']))
            weights = np.array([0.5, 0.3, 0.2])[:window_size]
            weights = weights / weights.sum()

            timbre_diff = 0
            for i in range(window_size):
                t1 = np.array(grad1['timbre'][-(i + 1)])
                t2 = np.array(grad2['timbre'][i])
                timbre_diff += weights[i] * np.mean((t1 - t2) ** 2)
            scores['timbre'] = np.sqrt(timbre_diff)

            pitch_diff = 0
            for i in range(window_size):
                p1 = np.array(grad1['pitches'][-(i + 1)])
                p2 = np.array(grad2['pitches'][i])
                pitch_diff += weights[i] * np.mean((p1 - p2) ** 2)
            scores['pitch'] = np.sqrt(pitch_diff)

            loudness_diff = abs(np.mean(grad1['loudness'][-window_size:]) -
                                np.mean(grad2['loudness'][:window_size]))
            scores['loudness'] = loudness_diff

            for key in scores:
                scores[key] = min(1.0, scores[key])

            return scores
        except Exception as e:
            logger.error(f"Error calculating compatibility: {e}")
            return {'timbre': 1.0, 'pitch': 1.0, 'loudness': 1.0}

    def analyze_rhythm_compatibility(self, segments1: List, segments2: List) -> float:
        """Analyze rhythm compatibility between segments with improved edge case handling."""
        try:
            def get_onset_pattern(segments):
                onsets = []
                current_time = 0
                for seg in segments:
                    if seg.get('confidence', 0) > 0.5:
                        onsets.append(current_time % 1.0)
                    current_time += seg['duration']
                return sorted(onsets) if onsets else None

            pattern1 = get_onset_pattern(segments1[-8:])
            pattern2 = get_onset_pattern(segments2[:8])

            if not pattern1 or not pattern2:
                return 0.5

            def pattern_similarity(p1, p2):
                if not p1 or not p2:
                    return 0.5

                try:
                    p1_max = max(p1) if max(p1) > 0 else 1.0
                    p2_max = max(p2) if max(p2) > 0 else 1.0

                    p1_normalized = np.array(p1) / p1_max if p1_max != 0 else np.zeros_like(p1)
                    p2_normalized = np.array(p2) / p2_max if p2_max != 0 else np.zeros_like(p2)

                    if np.all(p1_normalized == 0) or np.all(p2_normalized == 0):
                        return 0.5

                    distances = []
                    for i in p1_normalized:
                        if np.any(p2_normalized != 0):
                            distances.append(min(abs(i - j) for j in p2_normalized if j != 0))
                        else:
                            distances.append(1.0)

                    if not distances:
                        return 0.5

                    return 1 - (sum(distances) / len(distances))

                except Exception as e:
                    logger.warning(f"Error in pattern similarity calculation: {e}")
                    return 0.5

            return pattern_similarity(pattern1, pattern2)

        except Exception as e:
            logger.error(f"Error analyzing rhythm compatibility: {e}")
            return 0.5


    def display_track_results(self, tracks: List[Dict]):
        """Display track search results with improved formatting and additional info."""
        try:
            if not tracks:
                print("\nNo tracks found.")
                return

            print("\nSearch results:")
            print("-" * 80)

            for track in tracks:
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_name = key_names[track['key']] if track['key'] >= 0 else 'Unknown'
                mode_name = 'Major' if track['mode'] == 1 else 'Minor'

                print(f"{track['index']}. {track['name']} - {track['artist']}")
                print(f"   Album: {track['album']}")
                print(f"   Duration: {int(track['duration'] // 60)}:{int(track['duration'] % 60):02d}")
                print(f"   Tempo: {track['tempo']:.0f} BPM")
                print(f"   Key: {key_name} {mode_name}")
                print(f"   Time Signature: {track['time_signature']}/4")
                print("-" * 80)

        except Exception as e:
            logger.error(f"Error displaying track results: {e}")
            print("Error displaying results. Check logs for details.")

    def visualize_transition(self, segments1: List, segments2: List, title: str = "Transition Analysis"):
        """Visualize transition between tracks."""
        try:
            if not segments1 or not segments2:
                logger.warning("No segments to visualize")
                return

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(title)

            time1 = np.linspace(-self.SEGMENT_DURATION, 0, len(segments1))
            time2 = np.linspace(0, self.SEGMENT_DURATION, len(segments2))

            # Timbre plot
            timbre1 = [np.mean(seg['timbre']) for seg in segments1]
            timbre2 = [np.mean(seg['timbre']) for seg in segments2]
            ax1.plot(time1, timbre1, 'b-', label='Track 1', linewidth=2)
            ax1.plot(time2, timbre2, 'r-', label='Track 2', linewidth=2)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Timbre')
            ax1.set_title('Timbre Transition')
            ax1.legend()
            ax1.grid(True)

            # Pitch plot
            pitch1 = [np.mean(seg['pitches']) for seg in segments1]
            pitch2 = [np.mean(seg['pitches']) for seg in segments2]
            ax2.plot(time1, pitch1, 'b-', label='Track 1', linewidth=2)
            ax2.plot(time2, pitch2, 'r-', label='Track 2', linewidth=2)
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Pitch')
            ax2.set_title('Pitch Transition')
            ax2.legend()
            ax2.grid(True)

            # Loudness plot
            loudness1 = [seg['loudness_max'] for seg in segments1]
            loudness2 = [seg['loudness_max'] for seg in segments2]
            ax3.plot(time1, loudness1, 'b-', label='Track 1', linewidth=2)
            ax3.plot(time2, loudness2, 'r-', label='Track 2', linewidth=2)
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Loudness (dB)')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_title('Loudness Transition')
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error visualizing transition: {e}")

    def visualize_transition_heatmap(self, transition_data: Dict):
        """Create a heatmap visualization of the transition."""
        try:
            if not transition_data or 'end_segments' not in transition_data:
                logger.warning("No transition data to visualize")
                return

            end_segments = transition_data['end_segments']
            start_segments = transition_data['start_segments']

            def create_feature_matrix(segments, feature):
                if feature == 'timbre':
                    return np.array([seg['timbre'] for seg in segments])
                elif feature == 'pitches':
                    return np.array([seg['pitches'] for seg in segments])
                else:
                    return np.array([[seg['loudness_max']] for seg in segments])

            features = ['timbre', 'pitches', 'loudness']
            fig, axes = plt.subplots(len(features), 1, figsize=(12, 15))
            fig.suptitle('Transition Feature Heatmaps')

            for idx, feature in enumerate(features):
                matrix1 = create_feature_matrix(end_segments, feature)
                matrix2 = create_feature_matrix(start_segments, feature)

                combined = np.vstack([matrix1, matrix2])
                if feature != 'loudness':
                    combined = (combined - combined.min()) / (combined.max() - combined.min())

                im = axes[idx].imshow(combined.T, aspect='auto', cmap='coolwarm')
                axes[idx].axhline(y=matrix1.shape[1] - 0.5, color='white', linestyle='-')
                axes[idx].set_title(f'{feature.capitalize()} Transition')
                plt.colorbar(im, ax=axes[idx])

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error creating transition heatmap: {e}")

    def create_transition_report(self, track1: Dict, track2: Dict, transition_score: Dict) -> str:
        """Create a detailed report of the transition analysis."""
        try:
            report = f"""
Transition Analysis Report
=========================
From: {track1['name']} by {track1['artist']}
To: {track2['name']} by {track2['artist']}

Compatibility Scores
------------------
Timbre: {(1 - transition_score['timbre']) * 100:.1f}%
Pitch: {(1 - transition_score['pitch']) * 100:.1f}%
Loudness: {(1 - transition_score['loudness']) * 100:.1f}%
Tempo: {(1 - transition_score['tempo']) * 100:.1f}%
Key Compatibility: {(1 - transition_score['key_compatibility']) * 100:.1f}%
Rhythm: {(1 - transition_score['rhythm']) * 100:.1f}%

Track Details
------------
Track 1:
- Tempo: {track1['tempo']:.0f} BPM
- Key: {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][track1['key']]} {'Major' if track1['mode'] == 1 else 'Minor'}
- Duration: {int(track1['duration'] // 60)}:{int(track1['duration'] % 60):02d}

Track 2:
- Tempo: {track2['tempo']:.0f} BPM
- Key: {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][track2['key']]} {'Major' if track2['mode'] == 1 else 'Minor'}
- Duration: {int(track2['duration'] // 60)}:{int(track2['duration'] % 60):02d}

Overall Transition Quality: {(1 - transition_score['weighted_score']) * 100:.1f}%
"""
            return report

        except Exception as e:
            logger.error(f"Error creating transition report: {e}")
            return "Error generating report"

async def main():
    try:
        manager = SpotifyPlaybackManager()
        logger.info("Starting transition finder...")

        while True:
            query = input("\nEnter song name to search (or 'q' to quit): ")
            if query.lower() == 'q':
                break

            tracks = manager.search_tracks(query)
            if not tracks:
                print("No tracks found!")
                continue

            manager.display_track_results(tracks)

            while True:
                try:
                    choice = input("\nEnter the number of your choice (or 'r' to search again): ")
                    if choice.lower() == 'r':
                        break

                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(tracks):
                        selected_track = tracks[choice_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")

            if choice.lower() == 'r':
                continue

            print(f"\nSelected: {selected_track['name']} by {selected_track['artist']}")
            print("Finding best transitions... (this may take a moment)")

            best_intro, best_outro = await manager.find_best_transitions(selected_track['id'])

            if best_intro and best_outro:
                print("\nBest transitions found:")
                print(f"Intro track: {best_intro['name']} by {best_intro['artist']}")
                print(f"Transition smoothness: {(1 - best_intro['score']) * 100:.1f}%")

                print(f"\nMain track: {selected_track['name']} by {selected_track['artist']}")

                print(f"\nOutro track: {best_outro['name']} by {best_outro['artist']}")
                print(f"Transition smoothness: {(1 - best_outro['score']) * 100:.1f}%")

                # Visualize transitions if requested
                visualize = input("\nWould you like to see the transition visualizations? (y/n): ")
                if visualize.lower() == 'y':
                    if 'transition_data' in best_intro['details']:
                        manager.visualize_transition(
                            best_intro['details']['transition_data']['end_segments'],
                            best_intro['details']['transition_data']['start_segments'],
                            "Intro Transition"
                        )
                        manager.visualize_transition_heatmap(best_intro['details']['transition_data'])

                    if 'transition_data' in best_outro['details']:
                        manager.visualize_transition(
                            best_outro['details']['transition_data']['end_segments'],
                            best_outro['details']['transition_data']['start_segments'],
                            "Outro Transition"
                        )
                        manager.visualize_transition_heatmap(best_outro['details']['transition_data'])

            else:
                print("Could not find suitable transitions.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("An error occurred. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main())