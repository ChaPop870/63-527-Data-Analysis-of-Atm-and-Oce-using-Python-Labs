import calendar

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(3)

data = np.random.randint(27, size=(10, 12))

# Numer of sunny days in each year.
yearly_sunny_days = data.sum(axis=1)
start_year, end_year = 1990, 1999
years = np.arange(start_year, end_year + 1)

# Ave number of sunny days for each month.
ave_monthly_sunny_days = data.mean(axis=0)
months = np.array(calendar.month_abbr[1:])

# Seasonal share of sunny days.
winter_sunny_days = data[:, [11, 0, 1]].sum()
spring_sunny_days = data[:, [2, 3, 4]].sum()
summer_sunny_days = data[:, [5, 6, 7]].sum()
autumn_sunny_days = data[:, [8, 9, 10]].sum()

seasonal_sunny_days = [winter_sunny_days, spring_sunny_days, summer_sunny_days, autumn_sunny_days]


# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))

ax1.plot(years, yearly_sunny_days, color='blue')
ax1.set_xlabel("Year", fontsize=14)
ax1.set_ylabel("Number of Sunny days", fontsize=14)
ax1.set_title(f"Annual number of Sunny days at location ({start_year}-{end_year})", fontsize=16)
ax1.set_xlim(start_year, end_year)
ax1.set_ylim(100, 200)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(linestyle=':', alpha=0.3)

ax2.bar(months, ave_monthly_sunny_days, color='red')
ax2.set_xlabel("Month", fontsize=14)
ax2.set_ylabel(f"Average number of Sunny days per month", fontsize=14)
ax2.set_title(f"Average number of Sunny days per month at location ({start_year}-{end_year})", fontsize=16)
ax2.set_ylim(0, 20)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(linestyle=':', alpha=0.3)

ax3.pie(
    seasonal_sunny_days,
    labels=['Winter', 'Spring', 'Summer', 'Autumn'],
    colors=['cyan', 'limegreen', 'red', 'orange'],
    startangle=90,
    autopct=lambda p: f'{p:.1f}%',
    wedgeprops=dict(linewidth=0.5, edgecolor='black')
)
ax3.axis('equal')
ax3.set_title("Seasonal share of Sunny Days (1990-1999)", fontsize=16)

fig.suptitle(f"Summary plot for number of sunny days ({start_year}-{end_year})", y=1.0, fontsize=20)
plt.tight_layout()

plt.show()