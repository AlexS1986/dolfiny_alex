FROM dolfiny/dolfiny


# Set PYTHONPATH to include /home/utils
ENV PYTHONPATH="/home/utils:${PYTHONPATH}"

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]


