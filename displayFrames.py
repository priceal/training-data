"""
this script will display an array of frames.

v. 2021 05 24

"""

# set frames to display: xy = the particle dataframe and image = image number
displayframes_frames = frames
                     
# display parameters
columns = 10
rows = 10
auto_scale = True
######################################################################
# do not change code below this line
######################################################################
######################################################################
# analyze frames
num_frames, displayframes_width, displayframes_height = \
    displayframes_frames.shape       
figure_count = rows*columns
full_figs = int(num_frames/figure_count)
print('frame count = ', num_frames)
print('frame dimensions: {}'.format(displayframes_frames[0].shape))
print('display layout: {} rows X {} columns.'.format(columns,rows))
print( '{} frames per figure X {} figures'.format(figure_count, full_figs))
if full_figs*figure_count < num_frames:
    print( num_frames-figure_count*full_figs,'frames in last figure')

# create figures
last = 0
if num_frames >= figure_count:
    for figure in range(full_figs):

        first = figure*figure_count
        last = first + figure_count
        print( 'figure {}: frames {} to {}'.format(figure+1, first, last-1))
        pa.displayframes(displayframes_frames[first:last],rows,columns, \
                         rescale = auto_scale, count = first)

if last < num_frames:
    print( 'figure {}: frames {} to {}'.format(full_figs+1, last, num_frames-1))
    pa.displayframes(displayframes_frames[last:],rows,columns, \
                     rescale = auto_scale, count = last)

plt.show()
