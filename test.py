import vcf


vcf_read = vcf.Reader(open('chromosom22.vcf'))

vcf_write = vcf.Writer(open('chromaaaa22.vcf', 'w'), vcf_read)
for record in vcf_read:
    vcf_write.write_record(record.samples)
