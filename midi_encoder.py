import os
import re
import json
import numpy   as np
import math    as ma
import music21 as m21

THREE_DOTTED_BREVE = 15
THREE_DOTTED_32ND  = 0.21875

MIN_VELOCITY = 0
MAX_VELOCITY = 128

MIN_TEMPO = 24
MAX_TEMPO = 160

MAX_PITCH = 128

class MIDIEncoder:
    def __init__(self, datapath):
        self.text, self.vocab = self.load(datapath)

        self.vocab = list(set(self.vocab))
        self.vocab.sort()

        self.vocab_size = len(self.vocab)

        # Create dictionaries to support symbol to index conversion and vice-versa
        self.char2idx = { symb:i for i,symb in enumerate(self.vocab) }
        self.idx2char = { i:symb for i,symb in enumerate(self.vocab) }

    def load(self, datapath, sample_freq=4, piano_range=(33, 93), modulate_range=10, stretching_range=10):
        text = ""
        vocab = set()

        # Read every file in the given directory
        for file in os.listdir(datapath):
            file_path = os.path.join(datapath, file)
            file_extension = os.path.splitext(file_path)[1]

            # Check if it is not a directory and if it has either .midi or .mid extentions
            if os.path.isfile(file_path) and (file_extension == ".midi" or file_extension == ".mid"):
                print("Parsing midi file:", file_path)

                # Split datapath into dir and filename
                midi_dir = os.path.dirname(file_path)
                midi_name = os.path.basename(file_path).split(".")[0]

                # If txt version of the midi already exists, load data from it
                midi_txt_name = os.path.join(midi_dir, midi_name + ".txt")
                if(os.path.isfile(midi_txt_name)):
                    midi_fp = open(midi_txt_name, "r")
                    encoded_midi = midi_fp.read()
                else:
                    # Create a music21 stream and open the midi file
                    midi = m21.midi.MidiFile()
                    midi.open(file_path)
                    midi.read()
                    midi.close()

                    # Translate midi to stream of notes and chords
                    encoded_midi = self.midi2encoding(midi, sample_freq, piano_range, modulate_range, stretching_range)

                    if len(encoded_midi) > 0:
                        midi_fp = open(midi_txt_name, "w+")
                        midi_fp.write(encoded_midi)
                        midi_fp.flush()

                midi_fp.close()

                if len(encoded_midi) > 0:
                    words = set(encoded_midi.split(" "))
                    vocab = vocab | words

                    text += encoded_midi + " "

        return text[:-1], vocab

    def midi2encoding(self, midi, sample_freq, piano_range, modulate_range, stretching_range):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Get piano roll from midi stream
        piano_roll = self.midi2piano_roll(midi_stream, sample_freq, piano_range, modulate_range, stretching_range)

        # Get encoded midi from piano roll
        encoded_midi = self.piano_roll2encoding(piano_roll)

        return " ".join(encoded_midi)

    def piano_roll2encoding(self, piano_roll):
        # Transform piano roll into a list of notes in string format
        lastVelocity = -1
        lastDuration = -1.0

        final_encoding = {}

        perform_i = 0
        for version in piano_roll:
            version_encoding = []

            current_tempo = "t_120"
            for i in range(len(version)):

                # Time events are stored at the last row
                tempo_change = version[i,-1][0]
                if tempo_change != 0:
                    current_tempo = "t_" + str(int(tempo_change))

                # After every bar add last tempo mark.
                if i % 16 == 0:
                    version_encoding.append(current_tempo)

                # Process current time step of the piano_roll
                for j in range(len(version[i]) - 1):
                    duration = version[i,j][0]
                    velocity = int(version[i,j][1])

                    if velocity != 0 and velocity != lastVelocity:
                        version_encoding.append("v_" + str(velocity))

                    if duration != 0 and duration != lastDuration:
                        duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
                        version_encoding.append("d_" + duration_tuple.type + "_" + str(duration_tuple.dots))

                    if duration != 0 and velocity != 0:
                        version_encoding.append("n_" + str(j))

                    lastVelocity = velocity
                    lastDuration = duration

                # End of time step
                if version_encoding[-1][0] == "w":
                    # Increase wait by one
                    version_encoding[-1] = "w_" + str(int(version_encoding[-1].split("_")[1]) + 1)
                else:
                    version_encoding.append("w_1")

            # End of piece
            version_encoding.append("\n")

            # Check if this version of the MIDI is already added
            version_encoding_str = " ".join(version_encoding)
            if version_encoding_str not in final_encoding:
                final_encoding[version_encoding_str] = perform_i

            perform_i += 1

        return final_encoding.keys()

    @staticmethod
    def write(encoded_midi, path):
        # Base class checks if output path exists
        midi = MIDIEncoder.encoding2midi(encoded_midi)
        midi.open(path, "wb")
        midi.write()
        midi.close()

    @staticmethod
    def encoding2midi(note_encoding, ts_duration=0.25):
        notes = []

        velocity = 100
        duration = "16th"
        dots = 0

        ts = 0
        for note in note_encoding.split(" "):
            if len(note) == 0:
                continue

            elif note[0] == "w":
                wait_count = int(note.split("_")[1])
                ts += wait_count

            elif note[0] == "n":
                pitch = int(note.split("_")[1])
                note = m21.note.Note(pitch)
                note.duration = m21.duration.Duration(type=duration, dots=dots)
                note.offset = ts * ts_duration
                note.volume.velocity = velocity
                notes.append(note)

            elif note[0] == "d":
                duration = note.split("_")[1]
                dots = int(note.split("_")[2])

            elif note[0] == "v":
                velocity = int(note.split("_")[1])

            elif note[0] == "t":
                tempo = int(note.split("_")[1])

                if tempo > 0:
                    mark = m21.tempo.MetronomeMark(number=tempo)
                    mark.offset = ts * ts_duration
                    notes.append(mark)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)

    def midi_parse_notes(self, midi_stream, sample_freq):
        note_filter = m21.stream.filters.ClassFilter('Note')

        note_events = []
        for note in midi_stream.recurse().addFilter(note_filter):
            pitch    = note.pitch.midi
            duration = note.duration.quarterLength
            velocity = note.volume.velocity
            offset   = ma.floor(note.offset * sample_freq)

            note_events.append((pitch, duration, velocity, offset))

        return note_events

    def midi_parse_chords(self, midi_stream, sample_freq):
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        note_events = []
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                pitch    = pitch.midi
                duration = chord.duration.quarterLength
                velocity = chord.volume.velocity
                offset   = ma.floor(chord.offset * sample_freq)

                note_events.append((pitch, duration, velocity, offset))

        return note_events

    def midi_parse_metronome(self, midi_stream, sample_freq):
        metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

        time_events = []
        for metro in midi_stream.recurse().addFilter(metronome_filter):
            time = int(metro.number)
            offset = ma.floor(metro.offset * sample_freq)
            time_events.append((time, offset))

        return time_events

    def midi2notes(self, midi_stream, sample_freq, modulate_range):
        notes = []
        notes += self.midi_parse_notes(midi_stream, sample_freq)
        notes += self.midi_parse_chords(midi_stream, sample_freq)

        # Transpose the notes to all the keys in modulate_range
        return self.transpose_notes(notes, modulate_range)

    def midi2piano_roll(self, midi_stream, sample_freq, piano_range, modulate_range, stretching_range):
        # Calculate the amount of time steps in the piano roll
        time_steps = ma.floor(midi_stream.duration.quarterLength * sample_freq) + 1

        # Parse the midi file into a list of notes (pitch, duration, velocity, offset)
        transpositions = self.midi2notes(midi_stream, sample_freq, modulate_range)

        time_events = self.midi_parse_metronome(midi_stream, sample_freq)
        time_streches = self.strech_time(time_events, stretching_range)

        return self.notes2piano_roll(transpositions, time_streches, time_steps, piano_range)

    def notes2piano_roll(self, transpositions, time_streches, time_steps, piano_range):
        performances = []

        min_pitch, max_pitch = piano_range
        for t_ix in range(len(transpositions)):
            for s_ix in range(len(time_streches)):
                # Create piano roll with calcualted size.
                # Add one dimension to very entry to store velocity and duration.
                piano_roll = np.zeros((time_steps, MAX_PITCH + 1, 2))

                for note in transpositions[t_ix]:
                    pitch, duration, velocity, offset = note
                    if duration == 0.0:
                        continue

                    # Force notes to be inside the specified piano_range
                    pitch = self.__clamp_pitch(pitch, max_pitch, min_pitch)

                    piano_roll[offset, pitch][0] = self.__clamp_duration(duration)
                    piano_roll[offset, pitch][1] = self.discretize_value(velocity, bins=32, range=(MIN_VELOCITY, MAX_VELOCITY))

                for time_event in time_streches[s_ix]:
                    time, offset = time_event
                    piano_roll[offset, -1][0] = self.discretize_value(time, bins=100, range=(MIN_TEMPO, MAX_TEMPO))

                performances.append(piano_roll)

        return performances

    def transpose_notes(self, notes, modulate_range):
        transpositions = []

        # Modulate the piano_roll for other keys
        first_key = -ma.floor(modulate_range/2)
        last_key  =  ma.ceil(modulate_range/2)

        for key in range(first_key, last_key):
            notes_in_key = []
            for n in notes:
                pitch, duration, velocity, offset = n
                t_pitch = pitch + key
                notes_in_key.append((t_pitch, duration, velocity, offset))
            transpositions.append(notes_in_key)

        return transpositions

    def strech_time(self, time_events, stretching_range):
        streches = []

        # Modulate the piano_roll for other keys
        slower_time = -ma.floor(stretching_range/2)
        faster_time =  ma.ceil(stretching_range/2)

        # Modulate the piano_roll for other keys
        for t_strech in range(slower_time, faster_time):
            time_events_in_strech = []
            for t_ev in time_events:
                time, offset = t_ev
                s_time = time + 0.05 * t_strech * MAX_TEMPO
                time_events_in_strech.append((s_time, offset))
            streches.append(time_events_in_strech)

        return streches

    def discretize_value(self, val, bins, range):
        min_val, max_val = range

        val = int(max(min_val, val))
        val = int(min(val, max_val))

        bin_size = (max_val/bins)
        return ma.floor(val/bin_size) * bin_size

    def __clamp_pitch(self, pitch, max, min):
        while pitch < min:
            pitch += 12
        while pitch >= max:
            pitch -= 12
        return pitch

    def __clamp_duration(self, duration, max=THREE_DOTTED_BREVE, min=THREE_DOTTED_32ND):
        # Max duration is 3-dotted breve
        if duration > max:
            duration = max

        # min duration is 3-dotted breve
        if duration < min:
            duration = min

        duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
        if duration_tuple.type == "inexpressible":
            duration_clossest_type = m21.duration.quarterLengthToClosestType(duration)[0]
            duration = m21.duration.typeToDuration[duration_clossest_type]

        return duration
