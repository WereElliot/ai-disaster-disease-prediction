from django.contrib import admin

from .models import ClimateData, DiseaseIncidence, EmergencyReport, Prediction


@admin.register(EmergencyReport)
class EmergencyReportAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "county",
        "emergency_type",
        "severity",
        "status",
        "people_affected",
        "created_at",
    )
    list_filter = ("status", "severity", "emergency_type", "county")
    search_fields = ("county", "location_detail", "description", "reporter_name", "reporter_phone")
    readonly_fields = ("created_at", "updated_at", "reviewed_at")


admin.site.register(ClimateData)
admin.site.register(DiseaseIncidence)
admin.site.register(Prediction)
