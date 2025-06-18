import numpy as np
from scipy.io import wavfile
from TL import autocorrelate
import os
import copy

def est_voise(signal, seuil_autocorr=0.2, min_lag=20, max_lag=200):
    """
    Détecte si un segment est voisé.

    :param signal: le segment audio
    :param seuil_autocorr: seuil pour la détection de périodicité
    :param min_lag: lag minimum
    :param max_lag: lag maximum
    :return: bool (True si voisé)
    """
    autocorr = autocorrelate(signal, "nein")
    autocorr_part = autocorr[min_lag:max_lag+1]
    if len(autocorr_part) == 0:
        return False
    max_local = np.max(autocorr_part)
    return max_local > seuil_autocorr


def trouver_pitch(signal, fs, min_lag, max_lag):
    """
    Trouve le pitch (fréquence fondamentale) d'un segment voisé.

    :param signal: le segment audio
    :param fs: fréquence d'échantillonnage
    :param min_lag: lag minimum (samples)
    :param max_lag: lag maximum (samples)
    :return: float (fréquence en Hz), ou None si non voisé
    """
    if not est_voise(signal, min_lag=min_lag, max_lag=max_lag):
        return None
    autocorr = autocorrelate(signal, "no")
    autocorr_part = autocorr[min_lag:max_lag+1]
    if len(autocorr_part) == 0:
        return None
    lag = min_lag + np.argmax(autocorr_part)
    return fs / lag if lag > 0 else None


def freq_to_midi(freq):
    """
    Convertit une fréquence (Hz) en note MIDI.

    :param freq: fréquence (Hz)
    :return: float (note MIDI), ou None si fréquence invalide
    """
    if freq is None or freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440)))


def detecter_notes(data, fs, duree_tranche, min_pitch, max_pitch, duree_min_note):
    """
    Détecte les notes dans un signal audio.

    Il va analyser chaque tranche du son et détecter le pitch et la note, avant d'agréger tout cela dans un tableau.
    Ce tableau est ensuite réduit selon la note MIDI.
    Si plusieurs lignes de suite on la même note, nous supprimons les lignes suivant la première de la suite.

    :param data: signal audio (array)
    :param fs: fréquence d'échantillonnage (Hz)
    :param duree_tranche: durée d'une tranche (s)
    :param min_pitch: limite basse du pitch (Hz)
    :param max_pitch: limite haute du pitch (Hz)
    :param duree_min_note: durée minimale d'une note (s)
    :return: array (temps, fréquence, note MIDI)
    """
    n_ech_tranche = int(duree_tranche * fs)
    n_tranches = len(data) // n_ech_tranche
    min_lag = int(fs / max_pitch) if max_pitch > 0 else 20
    max_lag = int(fs / min_pitch) if min_pitch > 0 else 200

    # 1. Détection brute sur chaque fenêtre
    notes_brutes = []
    for i in range(n_tranches):
        debut = i * n_ech_tranche
        fin = debut + n_ech_tranche
        tranche = data[debut:fin]
        freq = trouver_pitch(tranche, fs, min_lag, max_lag).astype(float)
        if freq is not None and min_pitch <= freq <= max_pitch:
            note = freq_to_midi(freq)
            temps = i * duree_tranche
            notes_brutes.append([temps, freq, note])
        else:
            notes_brutes.append([i * duree_tranche, None, None])


    notes_fusionnees = []
    i = 0
    while i < len(notes_brutes):
        temps, freq, note = notes_brutes[i]
        j = i
        while j < len(notes_brutes) and notes_brutes[j][2] == note:
            j += 1
        temps_debut = notes_brutes[i][0]
        temps_fin = notes_brutes[j-1][0] + duree_tranche if j > i else temps_debut + duree_tranche
        duree = temps_fin - temps_debut
        if duree >= duree_min_note:
            notes_fusionnees.append([temps_debut, freq, note, duree])
        i = j

    return np.array(notes_fusionnees)


def analyser_audio(nom_fichier, duree_tranche, domaine_pitch, duree_min_note):
    """
    Analyse un fichier audio et retourne une matrice (temps, fréquence, note MIDI).
    :param nom_fichier: chemin du fichier .wav
    :param duree_tranche: durée d'une tranche temporelle (s)
    :param domaine_pitch: tuple (min_pitch, max_pitch) en Hz
    :param duree_min_note: durée minimale d'une note (s)
    :return: array (temps, fréquence, note MIDI)
    """
    fs, data = wavfile.read(nom_fichier)
    min_pitch, max_pitch = domaine_pitch
    return detecter_notes(data, fs, duree_tranche, min_pitch, max_pitch, duree_min_note)


if __name__ == "__main__":
    all_stats = []
    for wav in os.listdir("audios_TL"):
        print(wav)
        current_analyzed_audio = analyser_audio("audios_TL/" + wav, 0.02, [27, 4500], 0.02)

        cur_analyzed_audio_NAN = copy.deepcopy(current_analyzed_audio)
        cur_analyzed_audio_NAN[current_analyzed_audio == None] = np.nan
        cur_mean = np.nanmean(cur_analyzed_audio_NAN[:, 1], axis=0).astype(float)
        cur_var = np.nanvar(cur_analyzed_audio_NAN[:, 1], axis=0).astype(float)
        cur_audio_stats = [wav, cur_mean, cur_var]
        all_stats.append(cur_audio_stats)

        count_none = 0
        for delta in current_analyzed_audio:
            if delta[1] is None:
                count_none += 1
                print(f"\tt = {delta[0]}: fréquence = None")
            else:
                print(f"\tt = {round(delta[0], 2)} pendant {round(delta[3], 2)}s: fréquence = {round(delta[1].astype(float), 2)}Hz - node MIDI = {delta[2]}")
        print(f"{count_none}/{len(current_analyzed_audio)} de segments non voisés.")
        print("-----\n")

    print(np.array(all_stats))


