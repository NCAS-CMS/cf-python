import cfplot as cfp
import cf

f = cf.read("~/recipes_break/new/POLCOMS_WAM_ZUV_01_16012006.nc")
print(f)

# Get separate vector components
u = f[0]
v = f[1]
print(u)
print(v)

# First get rid of the ocean sigma coord size 1 axis
u = u.squeeze()
v = v.squeeze()

# Now we need to use some means to condense the u and v fields in the same way into
# having 1 time point, not 720 - for example we can just pick a time value out:
chosen_time = "2006-01-16 00:00:00"
v_1 = v.subspace(T=cf.dt(chosen_time))
u_1 = u.subspace(T=cf.dt(chosen_time))
v_1 = v_1.squeeze()
u_1 = u_1.squeeze()
print(u_1)
print(v_1)

# Are now in a plottable form! Let's give it a go:
### cfp.vect(u=u_1, v=v_1)
# Need to play around with the relative length and spacing of the vectors, using these paramters:
###cfp.vect(u=u_1, v=v_1, key_length=10, scale=50, stride=2)

# Note that there appear to be some really large vectors all pointing in the
# same direction which are spamming the plot. We need to remove these. By
# looking at the data we can see what these are and work out how to remove them:
print(u.data)
print(u[:10].data.array)

# ... shows more of the array

# Can see there are lots of -9999 values, seemingly used as a fill/placeholder value
# so we need to remove those so we can plot the menaingful vectors
# Apply steps to mask the -9999 fill values, which spam the plot, to x_1
u_2 = u_1.where(cf.lt(-9e+03), cf.masked)
v_2 = v_1.where(cf.lt(-9e+03), cf.masked)
print(u_2)
print(v_2)

# We can even plot the final field, effective wave height, as the
# background contour!
w = f[2]
w_1 = w.subspace(T=cf.dt(chosen_time))
# This field also needs masking for those data points.
w_2 = w_1.where(cf.lt(-9e+03), cf.masked)
print(w_2)
print(w_2, w_2[:10].data.array)

# Our final basic plot:
cfp.mapset(resolution="10m")  # makes UK coastline more high-res
cfp.gopen(file="irish-sea-currents.png")
# BTW ignore the warnings below - they aren't relevant.
cfp.vect(u=u_2, v=v_2, stride=2, scale=8, key_length=5)
cfp.levs(min=-5, max=5, step=0.5)
cfp.con(w_1, blockfill=True, lines=False)
cfp.gclose()

# Ideas for TODOs:
# investigate difference days (do this by changing the 'T=cf.dt("2006-01-16 00:00:00")') datetime
# values to different ones in the time coordinate data so you look at different days, or repace it
# with a collapse over some stat e.g. mean to show the mean over all the times,
# calculate divergence, calculate curl / relative voriticity, calculate absolute voriticity,
# explore the other dataset as well (that covers other dates/times) - you could compare the
# two to effectively compare the currents across different dates.
