"""
Name: Jack Sokhos
CS230:2
Data:McDonalds Store Reviews
"""

# app.py

import streamlit as st
st.set_page_config(page_title="McDonald's Reviews Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

# [PY3] Load CSV with pandas for correct parsing
def load_data(filepath='McDonald_s_Reviews.csv'):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df.to_dict('records')
        except UnicodeDecodeError:
            # try next encoding
            continue
        except Exception as e:
            # report other errors and stop
            st.error(f"Error loading data with encoding {enc}: {e}")
            return []
    # if all encodings fail
    st.error("Unable to decode CSV with utf-8, latin-1, or cp1252.")
    return []

# Load data
data_records = load_data()
if not data_records:
    st.stop()

# [DA1] Clean & convert fields
for record in data_records:
    # rating_numeric
    raw = record.get('rating', '')
    try:
        record['rating_numeric'] = int(str(raw).split()[0])
    except:
        record['rating_numeric'] = np.nan
    # rating_count
    try:
        record['rating_count'] = int(record.get('rating_count', 0))
    except:
        record['rating_count'] = 0
    # latitude & longitude
    try:
        record['latitude_float'] = float(record.get('latitude',
            record.get('latitude ', record.get('lat', np.nan))))
        record['longitude_float'] = float(record.get('longitude',
            record.get('longitude ', record.get('lon', np.nan))))
    except:
        record['latitude_float'] = np.nan
        record['longitude_float'] = np.nan

# Unique stores
unique_stores = []
for rec in data_records:
    addr = rec.get('store_address', '').strip()
    if addr and addr not in unique_stores:
        unique_stores.append(addr)
unique_stores.sort()

# Title
st.title("McDonald's Store Reviews Explorer")

# Sidebar filters
# Rating options
rating_options = []
for rec in data_records:
    r = rec['rating_numeric']
    if not np.isnan(r) and r not in rating_options:
        rating_options.append(r)
rating_options.sort()
selected_ratings = st.sidebar.multiselect(
    "Filter by Rating", rating_options, default=rating_options
)
# Reviewer count slider
max_count = 0
for rec in data_records:
    if rec['rating_count'] > max_count:
        max_count = rec['rating_count']
if max_count > 0:
    min_count = st.sidebar.slider(
        "Minimum Reviewer Rating Count", 0, int(max_count), 0
    )
else:
    min_count = 0
    st.sidebar.write("All reviewers have zero counts.")
# Store selector
to_options = ['All'] + unique_stores
selected_store = st.sidebar.selectbox(
    "Select Store Address", to_options
)
# Additional slider to pick a specific store for poor reviews
enabled = len(unique_stores) > 0
if enabled:
    loc_idx = st.sidebar.slider(
        "Select Store for Poor Reviews (by position)",
        1, len(unique_stores), 1
    )
    selected_poor_store = unique_stores[loc_idx - 1]
else:
    selected_poor_store = None

# [PY1] Filter by rating and count
def filter_reviews(records, ratings, min_val):
    filtered = []
    for rec in records:
        if rec['rating_numeric'] in ratings and rec['rating_count'] >= min_val:
            filtered.append(rec)
    return filtered
filtered = filter_reviews(data_records, selected_ratings, min_count)
# apply store filter
if selected_store != 'All':
    temp = []
    for rec in filtered:
        if rec.get('store_address', '').strip() == selected_store:
            temp.append(rec)
    filtered = temp

def filter_reviews(records, ratings, min_val):
    filtered = []
    for rec in records:
        if rec['rating_numeric'] in ratings and rec['rating_count'] >= min_val:
            filtered.append(rec)
    return filtered
filtered = filter_reviews(data_records, selected_ratings, min_count)
# apply store filter
if selected_store != 'All':
    temp = []
    for rec in filtered:
        if rec.get('store_address', '').strip() == selected_store:
            temp.append(rec)
    filtered = temp

st.markdown(f"**Showing {len(filtered)} reviews**")

# [PY2] Compute stats
def compute_stats(lst):
    count = len(lst)
    total = 0
    for rec in lst:
        total += rec['rating_numeric']
    avg = total / count if count else 0
    return count, avg
cnt, avg = compute_stats(filtered)
st.metric("Number of Reviews", cnt)
st.metric("Average Rating", f"{avg:.2f}")

# [VIZ1] Rating distribution histogram (1-5)
hist_data = []
for rec in filtered:
    hist_data.append(rec['rating_numeric'])
fig1, ax1 = plt.subplots()
ax1.hist(hist_data, bins=[0.5,1.5,2.5,3.5,4.5,5.5], edgecolor='black')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xlim(1,5)
ax1.set_title("Rating Distribution")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# [VIZ2] Reviews per Rating bar chart with labels
count_dict = {}
for r in rating_options:
    count_dict[r] = hist_data.count(r)
fig2, ax2 = plt.subplots()
bars2 = ax2.bar(count_dict.keys(), count_dict.values())
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h, str(int(h)), ha='center', va='bottom')
ax2.set_title("Reviews per Rating")
ax2.set_xlabel("Rating")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# [DA5] Summarize filtered by store, grp means 'group'
summary = {}
for rec in filtered:
    addr = rec.get('store_address', '').strip()
    if addr not in summary:
        summary[addr] = {'ratings': [], 'lat': rec['latitude_float'], 'lon': rec['longitude_float']}
    summary[addr]['ratings'].append(rec['rating_numeric'])
for grp in summary.values():
    tot = 0
    for val in grp['ratings']:
        tot += val
    grp['avg_rating'] = tot / len(grp['ratings']) if grp['ratings'] else 0
    grp['review_count'] = len(grp['ratings'])

# [VIZ3] Top 5 from filtered summary with bar labels
items = list(summary.items())
items.sort(key=lambda x: x[1]['avg_rating'], reverse=True)
top5 = items[:5]
names = []
avgs = []
for item in top5:
    names.append(item[0])
    avgs.append(item[1]['avg_rating'])
fig3, ax3 = plt.subplots()
bars3 = ax3.barh(names, avgs)
for bar in bars3:
    w = bar.get_width()
    ax3.text(w, bar.get_y() + bar.get_height()/2, f"{w:.2f}", ha='left', va='center')
ax3.set_title("Top 5 Stores by Avg. Rating")
ax3.set_xlabel("Avg. Rating")
st.pyplot(fig3)

# [MAP] Scatter map from filtered summary
to_plot = []
for addr, grp in summary.items():
    lat = grp['lat']
    lon = grp['lon']
    if not np.isnan(lat) and not np.isnan(lon):
        to_plot.append({'lat': lat, 'lon': lon, 'avg_rating': grp['avg_rating'], 'review_count': grp['review_count']})
if to_plot:
    view = pdk.ViewState(
        latitude=np.mean([pt['lat'] for pt in to_plot]),
        longitude=np.mean([pt['lon'] for pt in to_plot]), zoom=3
    )
    layer = pdk.Layer("ScatterplotLayer", to_plot, get_position=["lon","lat"], get_radius=20000, get_fill_color=[255,165,0,160], pickable=True)
    deck = pdk.Deck(initial_view_state=view, layers=[layer], tooltip={"text": "Avg: {avg_rating}\nCnt: {review_count}"})
    st.pydeck_chart(deck)

# [VIZ4] Top 5 Worst Reviews
st.subheader("Top 5 Worst Reviews")
worst_sorted = sorted(filtered, key=lambda x: x['rating_numeric'])[:5]
for idx, rec in enumerate(worst_sorted):
    loc = rec.get('store_address','').strip()
    rating = rec['rating_numeric']
    # attempt multiple possible text fields
    review_text = rec.get('review_text') or rec.get('ReviewText') or rec.get('review') or ''
    st.markdown(f"**Location:** {loc} — **Rating:** {rating}")
    # display the review in a read-only text area
    st.text_area(f"Review #{idx+1}", review_text, height=120, key=f"worst_{idx}")
    st.markdown("---")

# [VIZ5] Top 5 Best Reviews
st.subheader("Top 5 Best Reviews")
best_sorted = sorted(filtered, key=lambda x: x['rating_numeric'], reverse=True)[:5]
for idx, rec in enumerate(best_sorted):
    loc = rec.get('store_address','').strip()
    rating = rec['rating_numeric']
    review_text = rec.get('review_text') or rec.get('ReviewText') or rec.get('review') or ''
    st.markdown(f"**Location:** {loc} — **Rating:** {rating}")
    st.text_area(f"Review #{idx+1}", review_text, height=120, key=f"best_{idx}")
    st.markdown("---")
