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
print("Times are", v.construct("T").data.datetime_as_string)
chosen_time = "2006-01-15 23:30:00"  # 720 choices to choose from!
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


# Plot divergence in the background
div = cf.div_xy(u_2, v_2, radius="earth")


# Our final basic plot:
cfp.mapset(resolution="10m")  # makes UK coastline more high-res
cfp.gopen(file=f"irish-sea-currents-with-divergence-{chosen_time}.png")
cfp.cscale("ncl_default")
cfp.vect(u=u_2, v=v_2, stride=6, scale=3, key_length=1)
cfp.con(
    div,
    lines=False,
    title=(
        f"Depth-averaged Irish Sea currents at {chosen_time} "
        "with their divergence shown"
    )
)
cfp.gclose()
