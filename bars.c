#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <curses.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <unistd.h>

typedef char byte;

/*******************************************************************************
 * struct to store audio meta data
 * currently storing relevant information from WAV files.
 ******************************************************************************/
struct audio_meta {
  uint32_t sample_rate;
  int num_channels;
  int bits_per_sample;
  uint32_t size;
};

/*******************************************************************************
 * struct to point to audio data
 * linked list, where next points to the next channel
 ******************************************************************************/
struct audio {
  double *signal;
  struct audio *next;
};

void push_audio (struct audio **head, double *signal) {
  struct audio *p = malloc(sizeof(struct audio));
  p->signal = signal;
  p->next = *head;
  *head = p;
  return;
}

/*******************************************************************************
 * helper functions
 ******************************************************************************/
/* Standard C Function: Greatest Common Divisor */
/* from http://www.math.wustl.edu/~victor/mfmm/compaa/gcd.c */
unsigned int gcd (unsigned int a, unsigned int b)
{
  int c;
  while (a != 0) {
     c = a; a = b%a;  b = c;
  }
  return b;
}

long filesize(const char *name) {
  FILE *fl = fopen(name, "r");
  fseek(fl, 0, SEEK_END);
  long ret = ftell(fl);
  fclose(fl);
  return ret;
}

char* read_file_bytes(const char *name) {
  long len = filesize(name);
  char *ret = malloc(len);
  FILE *fl = fopen(name, "r");
  fseek(fl, 0, SEEK_SET);
  fread(ret, 1, len, fl);
  fclose(fl);
  return ret;
}

static double bytes_to_double(byte *first_byte, int num_bytes) {
  // little endian
  long ret = 0;
  for (int i = 0; i < num_bytes; ++i) {
    ret |= *(first_byte + i) << (i * 8);
  }
  // convert to range from -1 to (just below) 1
  return ret / pow(2, num_bytes * 8 - 1);
}

/*******************************************************************************
 * open a WAV file
 *
 * Input:
 * 1- char *filename :
 * 2- struct audio **ret : pointer to head of audio channels to be returned
 * 3- struct *audio_meta *meta: meta information to be returned
 ******************************************************************************/
void open_wav(char *filename, struct audio **ret, struct audio_meta *meta) {
  byte *wav = read_file_bytes(filename);
  meta->num_channels = *((uint16_t *) (wav + 22));
  meta->sample_rate = *((uint32_t *) (wav + 24));
  meta->bits_per_sample = *((uint16_t *) (wav + 34));
  int bytes_per_sample = meta->bits_per_sample / 8;
  int pos = 12;   // First Subchunk ID from 12 to 16
  while(!(wav[pos]==100 && wav[pos+1]==97 && wav[pos+2]==116 && wav[pos+3]==97)) pos++;
  pos += 4;
  meta->size = *((uint32_t *) (wav + pos));
  pos += 4;
  
  int samples_per_channel = meta->size / bytes_per_sample / meta->num_channels;

  for (int i = 0; i < meta->num_channels; ++i)
    push_audio(ret, malloc(sizeof(double) * samples_per_channel));

  byte *audio_data = wav + pos;
  // Write to double array/s:
  for (int audio_pos = 0, i = 0; audio_pos < meta->size; i++) {
    struct audio *channel = *ret;
    for (int j = 0; j < meta->num_channels; ++j) {
      channel->signal[i] = bytes_to_double(audio_data + audio_pos, bytes_per_sample);
      audio_pos += bytes_per_sample;
      channel = channel->next;
    }
  }
  free(wav);
  return;
}

/*******************************************************************************
 * perform a Fast Fourier Transform and bin the results
 *
 * Input:
 * 1- double *signal : pointer to audio data to transform
 * 2- int sig_len : length of signal
 * 3- fftw_plan : the FFT plan
 * 4- fftw_complex *in : pointer to FFT input
 * 5- fftw_complex *out : pointer to FFT output, should be same length as input
 * 6- int *bins: array of ints returned
 * 7- int num_bins : number of bins
 ******************************************************************************/
void fft_and_bin (double *signal, int sig_len,
                  fftw_plan p, fftw_complex *in, fftw_complex *out,
                  int *bins, int num_bins) {
  for (int i = 0; i < sig_len; i++) {
    __real__ in[i] = signal[i];
    __imag__ in[i] = 0.0;
  }
  fftw_execute(p);
  int bin_space = sig_len / 2 / num_bins; // this will miss the last
                                          // few bins, ok for now
  double *dbins = malloc(sizeof(double) * num_bins);
  for (int i = 0; i < num_bins; ++i) {
    dbins[i] = 0.0;
    for (int j = 0; j < bin_space; ++j) {
      dbins[i] += cabs(out[i * bin_space + j]) / (double) bin_space; // loses precision, ok for now
    }
    bins[i] = (int) dbins[i];
  }
  free(dbins);
  return;
}

/*******************************************************************************
 * logic of what to do for row j, given the bar is bar_height
 *
 * Input:
 * 1- int j : row height j
 * 2- int bar_height : bar height
 ******************************************************************************/
void paint_box (int j, int bar_height) {
  if (j <= bar_height) {
    attron(A_REVERSE);
  } else {
    attroff(A_REVERSE);
  }
  printw(" ");
  return;
}

/*******************************************************************************
 * update screen
 *
 * Input:
 * 1- int *bars : array to plot (bars[i] is the height of bin i)
 * 2- int num_bars : number of bars
 ******************************************************************************/
void refresh_bars (int *bars, int num_bars) {
  int scr_width, scr_height;
  getmaxyx(stdscr, scr_height, scr_width);
  for (int i = 0; i < num_bars; ++i) {
    for (int j = 0; j < scr_height; ++j) {
      move(scr_height - j - 1, i);
      paint_box(j, bars[i]);
    }
  }
  refresh();
  return;
}

/*******************************************************************************
 * calculate the refresh rate and the number of samples on which to FFT
 *
 * Input:
 * 1- int sample_rate : number of samples per second
 * 2- int *refresh_rate : return the screen refresh rate (in milliseconds)
 * 3- int *local_len : return the number of samples in each refresh
 ******************************************************************************/
void calc_rates (int sample_rate, int *refresh_rate, int *local_len) {
  int g = gcd(1000, sample_rate);
  *local_len = sample_rate / g;
  *refresh_rate = 1000 / g;
  while (*local_len < 256) {
    *local_len *= 2;
    *refresh_rate *= 2;
  }
  return;
}

/*******************************************************************************
 * visualize an audio stream
 *
 * Input:
 * 1- double *signal : PCM array
 * 2- int sig_len : length of signal
 * 3- int local_len : how many samples in each refresh of the screen (ie, the
 * number of samples on which to FFT each time)
 * 4- int refresh_rate : number of milliseconds between each screen refresh
 ******************************************************************************/
void visualize (double *signal, int sig_len, int local_len, int refresh_rate) {
  fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * local_len);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * local_len);
  fftw_plan p = fftw_plan_dft_1d(local_len, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  int pos = 0;
  initscr();
  curs_set(0);
  noecho();
  int scr_width, scr_height, num_bars;
  int *bars = malloc(sizeof(int));
  for (int pos = 0; pos < sig_len; pos += local_len) {
    getmaxyx(stdscr, scr_height, scr_width);
    if (num_bars != scr_width) {
      bars = realloc(bars, sizeof(int) * scr_width);
      num_bars = scr_width;
    }
    fft_and_bin(signal + pos, local_len, p, in, out, bars, num_bars);
    refresh_bars(bars, num_bars);
    usleep(1000 * refresh_rate);
  }
  endwin();
  fftw_free(in); fftw_free(out);
  free(bars);
  return;
}

/*******************************************************************************
 * main - glue
 *
 * Usage: ./bars filename.wav
 ******************************************************************************/
int main(int argc, char *argv[]) {
  struct audio_meta meta;
  struct audio *audio;
  open_wav(argv[1], &audio, &meta);
  int refresh_rate, local_len;
  calc_rates(meta.sample_rate, &refresh_rate, &local_len);
  visualize(audio->signal, meta.size / meta.num_channels / meta.bits_per_sample * 8,
            local_len, refresh_rate);
  return 0;
}
