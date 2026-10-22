FROM sigmod2024id403/pg13_hacked_for_ce_benchmark

USER postgres

RUN { \
      echo "listen_addresses = '*'"; \
      echo "shared_buffers = 12GB"; \
      echo "work_mem = 2GB"; \
      echo "effective_cache_size = 32GB"; \
      echo "geqo_threshold = 18"; \
    } >> /var/lib/pgsql/13.1/data/postgresql.conf

COPY --chown=postgres:postgres stats.sql indexes.sql init_stats_db.sh /opt/stats/

EXPOSE 5432
CMD ["postgres", "-D", "/var/lib/pgsql/13.1/data"]
