from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
import json
from datetime import datetime

from core.models import EmergencyReport
from core.services import PredictionService

prediction_service = PredictionService()
CLIMATE_REGION_OPTIONS = prediction_service.get_available_climate_regions()

KENYA_COUNTIES = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu",
    "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho",
    "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale",
    "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", "Marsabit",
    "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", "Nakuru",
    "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", "Samburu",
    "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", "Trans Nzoia",
    "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot",
]


def _community_page_context(title, active_page, **extra):
    context = {
        "title": title,
        "active_page": active_page,
        "current_year": datetime.now().year,
        "climate_regions": CLIMATE_REGION_OPTIONS,
        "search_value": "",
    }
    context.update(extra)
    return context


def _emergency_report_form_data(request):
    return {
        "reporter_name": request.POST.get("reporter_name", "").strip(),
        "reporter_phone": request.POST.get("reporter_phone", "").strip(),
        "county": request.POST.get("county", "").strip(),
        "location_detail": request.POST.get("location_detail", "").strip(),
        "emergency_type": request.POST.get("emergency_type", "").strip(),
        "severity": request.POST.get("severity", "").strip() or "medium",
        "description": request.POST.get("description", "").strip(),
        "people_affected": request.POST.get("people_affected", "").strip(),
    }


def _emergency_report_context(form_data=None):
    return _community_page_context(
        "FutureGuard Emergency Report",
        "emergency",
        county_options=KENYA_COUNTIES,
        emergency_type_options=EmergencyReport.EMERGENCY_TYPES,
        severity_options=EmergencyReport.SEVERITY_LEVELS,
        form_data=form_data or {
            "reporter_name": "",
            "reporter_phone": "",
            "county": "",
            "location_detail": "",
            "emergency_type": "",
            "severity": "medium",
            "description": "",
            "people_affected": "",
        },
        submitted=False,
    )


def home(request):
    context = _community_page_context(
        "FutureGuard Community Home",
        "home",
        search_value=request.GET.get("region", "").strip(),
    )
    return render(request, "community_dashboard/home.html", context)


def predictions(request):
    context = _community_page_context(
        "FutureGuard Community Predictions",
        "predictions",
        search_value=request.GET.get("region", "").strip(),
        initial_region=request.GET.get("region", "").strip(),
    )
    return render(request, "community_dashboard/predictions.html", context)


def emergency_report(request):
    context = _emergency_report_context()
    context["search_value"] = request.GET.get("region", "").strip()

    if request.method == "POST":
        form_data = _emergency_report_form_data(request)
        context = _emergency_report_context(form_data=form_data)
        errors = []

        if not form_data["county"]:
            errors.append("County is required.")
        if not form_data["emergency_type"]:
            errors.append("Emergency type is required.")
        if not form_data["severity"]:
            errors.append("Severity is required.")
        if not form_data["description"]:
            errors.append("Please describe the emergency.")

        people_affected = 0
        if form_data["people_affected"]:
            try:
                people_affected = max(0, int(form_data["people_affected"]))
            except ValueError:
                errors.append("People affected must be a whole number.")

        if errors:
            context["form_error"] = " ".join(errors)
            return render(request, "community_dashboard/emergency_report.html", context)

        report = EmergencyReport.objects.create(
            reporter_name=form_data["reporter_name"] or "Anonymous Community Member",
            reporter_phone=form_data["reporter_phone"],
            county=form_data["county"],
            location_detail=form_data["location_detail"],
            emergency_type=form_data["emergency_type"],
            severity=form_data["severity"],
            description=form_data["description"],
            people_affected=people_affected,
        )

        return redirect(
            f"{reverse('community_dashboard:emergency_report')}?submitted=1&report={report.id}"
        )

    if request.GET.get("submitted") == "1":
        context["submitted"] = True
        context["submitted_reference"] = request.GET.get("report", "")

    return render(request, "community_dashboard/emergency_report.html", context)


@csrf_exempt
def api_predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            location = (
                data.get("location")
                or data.get("region")
                or data.get("county")
                or ""
            ).strip()
            
            if not location:
                return JsonResponse({'error': 'Climate region required'}, status=400)

            resolved_location = prediction_service.resolve_location(location)
            if not resolved_location:
                return JsonResponse(
                    {'error': 'Enter a supported Kenya climatic region'},
                    status=400,
                )
            
            predictions = prediction_service.predict_all(resolved_location)
            return JsonResponse(predictions)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def api_action_plan(request):
    """Generate action plan based on predictions - Addresses ALL risks with full Kiswahili translations"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            county = data.get('county')
            predictions = data.get('predictions', {})
            language = data.get('language', 'en')
            
            print(f"Generating action plan for {county} in {language}")
            print(f"Predictions received: {predictions}")

            risk_catalog = {
                'malaria': {'label_en': 'Malaria', 'label_sw': 'Malaria', 'category': 'disease'},
                'cholera': {'label_en': 'Cholera', 'label_sw': 'Kipindupindu', 'category': 'disease'},
                'dengue': {'label_en': 'Dengue', 'label_sw': 'Dengue', 'category': 'disease'},
                'floods': {'label_en': 'Floods', 'label_sw': 'Mafuriko', 'category': 'disaster'},
                'drought': {'label_en': 'Drought', 'label_sw': 'Ukame', 'category': 'disaster'},
            }
            risk_levels_en = {
                'critical': 'Critical',
                'high': 'High',
                'medium': 'Medium',
                'low': 'Low',
            }
            risk_levels_sw = {
                'critical': 'Hatari Kubwa Sana',
                'high': 'Hatari Kubwa',
                'medium': 'Hatari ya Kati',
                'low': 'Hatari Ndogo',
            }
            valid_prediction_items = {
                disease: info
                for disease, info in predictions.items()
                if disease in risk_catalog and isinstance(info, dict)
            }
            
            # Identify ALL risks (critical, high, medium)
            critical_risks = []
            high_risks = []
            medium_risks = []
            
            for disease, info in valid_prediction_items.items():
                risk = info.get('risk', 'low')
                if risk == 'critical':
                    critical_risks.append(disease)
                elif risk == 'high':
                    high_risks.append(disease)
                elif risk == 'medium':
                    medium_risks.append(disease)
            
            print(f"Critical risks: {critical_risks}")
            print(f"High risks: {high_risks}")
            print(f"Medium risks: {medium_risks}")
            
            # Risk name translations for Kiswahili
            risk_names_sw = {
                disease: details['label_sw']
                for disease, details in risk_catalog.items()
            }
            
            # Action templates with ALL levels for each disease/disaster
            action_templates = {
                'malaria': {
                    'en': {
                        'critical': [
                            "EMERGENCY: Activate rapid response teams for indoor residual spraying",
                            "Distribute mosquito nets to ALL households in affected areas",
                            "Set up temporary testing centers in high-traffic areas",
                            "Mobilize community health workers for active case finding"
                        ],
                        'high': [
                            "Distribute mosquito nets to households with pregnant women and children under 5",
                            "Drain standing water around homes and communities",
                            "Report any fever cases to the nearest health facility immediately",
                            "Conduct community awareness campaigns on malaria prevention"
                        ],
                        'medium': [
                            "Ensure mosquito nets are properly used at night",
                            "Clear bushes and remove stagnant water around homes",
                            "Monitor for fever symptoms in children",
                            "Check health facility medicine stock levels"
                        ],
                        'low': [
                            "Continue using mosquito nets",
                            "Maintain good hygiene practices",
                            "Stay informed about malaria prevention"
                        ]
                    },
                    'sw': {
                        'critical': [
                            "HARAKA: Amua timu za kukabiliana na dharura kwa kunyunyizia dawa ndani ya nyumba",
                            "Sambaza vyandarua kwa KAYA ZOTE katika maeneo yaliyoathirika",
                            "Anzisha vituo vya kupimia katika maeneo yenye watu wengi",
                            "Hamasa wafanyakazi wa afya ya jamii kutafuta wagonjwa"
                        ],
                        'high': [
                            "Sambaza vyandarua kwa familia zenye wajawazito na watoto chini ya miaka 5",
                            "Fukuza maji yaliyotuama kwenye nyua na jamii",
                            "Ripoti kesi zozote za homa kwenye kituo cha afya kilicho karibu mara moja",
                            "Fanya kampeni za uhamasishaji wa jamii kuhusu kinga ya malaria"
                        ],
                        'medium': [
                            "Hakikisha vyandarua vinatumiwa vizuri usiku",
                            "Ondoa vichaka na maji yaliyotuama karibu na nyumba",
                            "Fuatilia dalili za homa kwa watoto",
                            "Kagua upatikanaji wa dawa kwenye vituo vya afya"
                        ],
                        'low': [
                            "Endelea kutumia vyandarua",
                            "Dumisha usafi wa mazingira",
                            "Jua habari za kinga ya malaria"
                        ]
                    }
                },
                'cholera': {
                    'en': {
                        'critical': [
                            "EMERGENCY: Activate cholera treatment centers immediately",
                            "Ban public gatherings in affected zones",
                            "Set up oral rehydration points in affected areas",
                            "Emergency water treatment and chlorination of all water sources"
                        ],
                        'high': [
                            "Ensure all drinking water is boiled or treated with chlorine",
                            "Promote handwashing with soap before eating and after using the toilet",
                            "Avoid eating raw or undercooked food, especially seafood",
                            "Report any diarrhea cases to health facility immediately"
                        ],
                        'medium': [
                            "Check water sources for contamination",
                            "Practice proper food handling and storage",
                            "Report any diarrhea cases to health facility",
                            "Intensify hygiene promotion in communities"
                        ],
                        'low': [
                            "Maintain handwashing habits",
                            "Drink safe water",
                            "Keep food covered"
                        ]
                    },
                    'sw': {
                        'critical': [
                            "HARAKA: Anzisha vituo vya matibabu ya kipindupindu mara moja",
                            "Piga marufuku mikusanyiko ya umma katika maeneo yaliyoathirika",
                            "Weka vituo vya kutoa maji ya upungufu wa maji mwilini",
                            "Safisha maji ya dharura kwa kemikali kwenye vyanzo vyote vya maji"
                        ],
                        'high': [
                            "Hakikisha maji yote ya kunywa yanachemshwa au kutibiwa na kemikali",
                            "Kuza usafi wa kunawa mikono kwa sabuni kabla ya kula na baada ya kutumia choo",
                            "Epuka kula vyakula vibichi au visivyoiva, hasa vyakula vya baharini",
                            "Ripoti kesi zozote za kuhara kwenye kituo cha afya mara moja"
                        ],
                        'medium': [
                            "Kagua vyanzo vya maji kwa uchafuzi",
                            "Shughulikia vyakula kwa usafi na uhifadhi sahihi",
                            "Ripoti kesi zozote za kuhara kwenye kituo cha afya",
                            "Zidisha uhamasishaji wa usafi miongoni mwa jamii"
                        ],
                        'low': [
                            "Endelea kuosha mikono",
                            "Kunywa maji salama",
                            "Funika vyakula"
                        ]
                    }
                },
                'dengue': {
                    'en': {
                        'critical': [
                            "EMERGENCY: Activate vector control teams for intensive spraying",
                            "Conduct door-to-door mosquito breeding site elimination",
                            "Set up dengue fever testing centers",
                            "Issue community-wide alert about dengue symptoms"
                        ],
                        'high': [
                            "Eliminate mosquito breeding sites by covering water storage containers",
                            "Use mosquito repellent and sleep under treated nets",
                            "Seek medical care if you develop high fever with severe headache",
                            "Remove standing water from flower pots, tires, and containers"
                        ],
                        'medium': [
                            "Cover water tanks and containers",
                            "Use window screens or mosquito nets",
                            "Remove standing water from flower pots and tires",
                            "Use mosquito repellent when outdoors"
                        ],
                        'low': [
                            "Maintain clean surroundings",
                            "Use mosquito repellent when outdoors",
                            "Check for stagnant water weekly"
                        ]
                    },
                    'sw': {
                        'critical': [
                            "HARAKA: Amua timu za kudhibiti mbu kwa kunyunyizia dawa",
                            "Fanya ziara nyumba kwa nyumba kuondoa maeneo ya kuzaliana mbu",
                            "Anzisha vituo vya kupima homa ya dengue",
                            "Toa tahadhari kwa jamii kuhusu dalili za dengue"
                        ],
                        'high': [
                            "Ondoa maeneo ya kuzaliana mbu kwa kufunika vyombo vya kuhifadhi maji",
                            "Tumia dawa za kufukuza mbu na lala chini ya vyandarua",
                            "Tafuta matibabu ikiwa una homa kali na maumivu makali ya kichwa",
                            "Ondoa maji yaliyotuama kwenye vyungu vya maua, matairi na vyombo"
                        ],
                        'medium': [
                            "Funika matangi na vyombo vya maji",
                            "Tumia vyandarua kwenye madirisha au vyandarua",
                            "Ondoa maji yaliyotuama kwenye vyungu vya maua na matairi",
                            "Tumia dawa ya kufukuza mbu ukiwa nje"
                        ],
                        'low': [
                            "Dumisha usafi wa mazingira",
                            "Tumia dawa ya kufukuza mbu ukiwa nje",
                            "Kagua maji yaliyotuama kila wiki"
                        ]
                    }
                },
                'floods': {
                    'en': {
                        'critical': [
                            "EMERGENCY: Evacuate to designated safe zones immediately",
                            "Move to higher ground if you live in low-lying areas",
                            "Store important documents and valuables in waterproof containers",
                            "Do NOT attempt to walk or drive through flood waters"
                        ],
                        'high': [
                            "Move to higher ground if you live in low-lying areas",
                            "Store important documents and valuables in waterproof containers",
                            "Avoid walking or driving through flood waters",
                            "Keep emergency supplies ready (food, water, medicine)"
                        ],
                        'medium': [
                            "Clear drainage channels around your home",
                            "Prepare sandbags for doorways",
                            "Keep emergency supplies ready",
                            "Monitor weather forecasts regularly"
                        ],
                        'low': [
                            "Monitor weather forecasts",
                            "Ensure gutters and drains are clear",
                            "Have an emergency contact list ready"
                        ]
                    },
                    'sw': {
                        'critical': [
                            "HARAKA: Hamia kwenye maeneo salama yaliyotengwa mara moja",
                            "Hamia maeneo ya juu ikiwa unaishi katika maeneo ya chini",
                            "Hifadhi hati muhimu na vitu vya thamani kwenye vyombo visivyopitisha maji",
                            "USIJARIBU kutembea au kuendesha gari kwenye maji ya mafuriko"
                        ],
                        'high': [
                            "Hamia maeneo ya juu ikiwa unaishi katika maeneo ya chini",
                            "Hifadhi hati muhimu na vitu vya thamani kwenye vyombo visivyopitisha maji",
                            "Epuka kutembea au kuendesha gari kwenye maji ya mafuriko",
                            "Weka vifaa vya dharura tayari (chakula, maji, dawa)"
                        ],
                        'medium': [
                            "Safisha mifereji ya maji karibu na nyumba yako",
                            "Andaa magunia ya mchanga kwenye milango",
                            "Weka vifaa vya dharura tayari",
                            "Fuatilia utabiri wa hali ya hewa mara kwa mara"
                        ],
                        'low': [
                            "Fuatilia utabiri wa hali ya hewa",
                            "Hakikisha mifereji imesafishwa",
                            "Kuwa na orodha ya mawasiliano ya dharura"
                        ]
                    }
                },
                'drought': {
                    'en': {
                        'critical': [
                            "EMERGENCY: Activate water trucking to affected areas",
                            "Distribute food supplements to vulnerable households",
                            "Set up nutrition screening for children under 5",
                            "Monitor livestock conditions daily"
                        ],
                        'high': [
                            "Conserve water by fixing leaks and reducing usage",
                            "Store water in clean containers for future use",
                            "Monitor children and elderly for signs of dehydration",
                            "Plant drought-resistant crops"
                        ],
                        'medium': [
                            "Use water efficiently in households",
                            "Collect rainwater when available",
                            "Plant drought-resistant crops",
                            "Monitor water levels regularly"
                        ],
                        'low': [
                            "Practice water conservation",
                            "Monitor water levels",
                            "Stay hydrated"
                        ]
                    },
                    'sw': {
                        'critical': [
                            "HARAKA: Anzisha usafirishaji wa maji kwa lori kwenye maeneo yaliyoathirika",
                            "Sambaza virutubisho vya chakula kwa familia zilizo katika hatari",
                            "Anzisha uchunguzi wa lishe kwa watoto chini ya miaka 5",
                            "Fuatilia hali ya mifugo kila siku"
                        ],
                        'high': [
                            "Hifadhi maji kwa kuziba uvujaji na kupunguza matumizi",
                            "Hifadhi maji kwenye vyombo safi kwa matumizi ya baadaye",
                            "Fuatilia watoto na wazee kwa dalili za upungufu wa maji mwilini",
                            "Panda mazao yanayostahimili ukame"
                        ],
                        'medium': [
                            "Tumia maji kwa ufanisi nyumbani",
                            "Kusanya maji ya mvua inaponyesha",
                            "Panda mazao yanayostahimili ukame",
                            "Fuatilia kiwango cha maji mara kwa mara"
                        ],
                        'low': [
                            "Zoea kuhifadhi maji",
                            "Fuatilia kiwango cha maji",
                            "Kunywa maji ya kutosha"
                        ]
                    }
                }
            }
            
            # Helper function to get actions for a disease based on risk level
            def get_actions_for_disease(disease, risk_level):
                if disease in action_templates:
                    templates = action_templates[disease][language]
                    if risk_level == 'critical' and 'critical' in templates:
                        return templates['critical']
                    elif risk_level == 'high' and 'high' in templates:
                        return templates['high']
                    elif risk_level == 'medium' and 'medium' in templates:
                        return templates['medium']
                    else:
                        return templates.get('low', [])
                return []
            
            # Build action plan - COLLECT ACTIONS FROM ALL RISKS
            immediate_actions = []
            weekly_actions = []
            risk_summary_items = []
            
            # Add actions for critical risks (highest priority)
            for risk in critical_risks:
                actions = get_actions_for_disease(risk, 'critical')
                immediate_actions.extend(actions[:3])
                # TRANSLATED weekly action for critical risks
                if language == 'en':
                    weekly_actions.append(f"URGENT: Daily surveillance and immediate reporting for {risk}")
                else:
                    risk_sw = risk_names_sw.get(risk, risk)
                    weekly_actions.append(f"HARAKA: Ufuatiliaji wa kila siku na ripoti ya haraka kwa {risk_sw}")
                # Add to risk summary
                risk_name = risk.capitalize()
                if language == 'sw':
                    risk_name = risk_names_sw.get(risk, risk.capitalize())
                risk_summary_items.append(f"🔴 CRITICAL: {risk_name}")
            
            # Add actions for high risks
            for risk in high_risks:
                actions = get_actions_for_disease(risk, 'high')
                immediate_actions.extend(actions[:2])
                # TRANSLATED weekly action for high risks
                if language == 'en':
                    weekly_actions.append(f"Monitor {risk} cases weekly and report to health authorities")
                else:
                    risk_sw = risk_names_sw.get(risk, risk)
                    weekly_actions.append(f"Fuatilia kesi za {risk_sw} kila wiki na ripoti kwa mamlaka ya afya")
                # Add to risk summary
                risk_name = risk.capitalize()
                if language == 'sw':
                    risk_name = risk_names_sw.get(risk, risk.capitalize())
                risk_summary_items.append(f"🟠 HIGH: {risk_name}")
            
            # Add actions for medium risks
            for risk in medium_risks:
                actions = get_actions_for_disease(risk, 'medium')
                weekly_actions.extend(actions[:2])
                # TRANSLATED weekly action for medium risks (optional)
                if language == 'en':
                    weekly_actions.append(f"Maintain surveillance for {risk}")
                else:
                    risk_sw = risk_names_sw.get(risk, risk)
                    weekly_actions.append(f"Dumisha ufuatiliaji wa {risk_sw}")
                # Add to risk summary
                risk_name = risk.capitalize()
                if language == 'sw':
                    risk_name = risk_names_sw.get(risk, risk.capitalize())
                risk_summary_items.append(f"🟡 MEDIUM: {risk_name}")
            
            # Remove duplicates while preserving order
            immediate_actions = list(dict.fromkeys(immediate_actions))
            weekly_actions = list(dict.fromkeys(weekly_actions))
            
            # Limit to reasonable numbers
            immediate_actions = immediate_actions[:6]
            weekly_actions = weekly_actions[:5]
            
            # Create risk summary text
            if risk_summary_items:
                risk_summary = "\n".join(risk_summary_items)
            else:
                if language == 'en':
                    risk_summary = "✅ No immediate risks detected"
                else:
                    risk_summary = "✅ Hakuna hatari iliyogunduliwa"
            
            # If no risks detected, show general prevention
            if not critical_risks and not high_risks and not medium_risks:
                if language == 'en':
                    immediate_actions = [
                        "Continue routine prevention measures",
                        "Stay informed through local health bulletins",
                        "Maintain good hygiene practices",
                        "Check water sources regularly"
                    ]
                    weekly_actions = [
                        "Participate in community health education sessions",
                        "Monitor local health alerts",
                        "Share prevention information with neighbors"
                    ]
                else:
                    immediate_actions = [
                        "Endelea na hatua za kawaida za kuzuia",
                        "Pata habari kupitia matangazo ya afya ya eneo lako",
                        "Dumisha usafi wa mazingira",
                        "Kagua vyanzo vya maji mara kwa mara"
                    ]
                    weekly_actions = [
                        "Shiriki katika vikao vya elimu ya afya ya jamii",
                        "Fuatilia tahadhari za afya za eneo lako",
                        "Shiriki habari za kuzuia na majirani"
                    ]
            
            # Ongoing actions
            if language == 'en':
                ongoing_actions = [
                    "Participate in community health education sessions",
                    "Maintain good hygiene practices",
                    "Stay informed through local health bulletins",
                    "Report any unusual health events to health authorities"
                ]
                emergency_contacts = [
                    {"name": "County Health Office", "phone": "0800-720-720"},
                    {"name": "Ambulance", "phone": "911"},
                    {"name": "Red Cross", "phone": "020-604-1234"},
                    {"name": "WHO Emergency Hotline", "phone": "+254-20-271-1234"}
                ]
            else:
                ongoing_actions = [
                    "Shiriki katika vikao vya elimu ya afya ya jamii",
                    "Dumisha usafi wa mazingira",
                    "Jua habari kupitia matangazo ya afya ya eneo lako",
                    "Ripoti matukio yoyote ya kiafya yasiyo ya kawaida kwa viongozi wa afya"
                ]
                emergency_contacts = [
                    {"name": "Afya ya Kaunti", "phone": "0800-720-720"},
                    {"name": "Ambulensi", "phone": "911"},
                    {"name": "Msalaba Mwekundu", "phone": "020-604-1234"},
                    {"name": "Nambari ya Dharura ya WHO", "phone": "+254-20-271-1234"}
                ]

            hazard_sections = []
            for disease in ['malaria', 'cholera', 'dengue', 'floods', 'drought']:
                info = valid_prediction_items.get(disease)
                if not info:
                    continue
                risk_level = info.get('risk', 'low')
                hazard_sections.append(
                    {
                        'id': disease,
                        'label': (
                            risk_catalog[disease]['label_sw']
                            if language == 'sw'
                            else risk_catalog[disease]['label_en']
                        ),
                        'category': risk_catalog[disease]['category'],
                        'category_label': (
                            'Magonjwa' if risk_catalog[disease]['category'] == 'disease' else 'Majanga'
                        ) if language == 'sw' else (
                            'Disease' if risk_catalog[disease]['category'] == 'disease' else 'Disaster'
                        ),
                        'risk_level': risk_level,
                        'risk_label': (
                            risk_levels_sw.get(risk_level, risk_level)
                            if language == 'sw'
                            else risk_levels_en.get(risk_level, risk_level.capitalize())
                        ),
                        'confidence': info.get('confidence'),
                        'actions': get_actions_for_disease(disease, risk_level)[:4],
                    }
                )
            
            action_plan = {
                'county': county,
                'climate_region': predictions.get('climate_region', county),
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'risk_summary': risk_summary,
                'critical_risks': critical_risks,
                'high_risks': high_risks,
                'medium_risks': medium_risks,
                'immediate_actions': immediate_actions,
                'weekly_actions': weekly_actions,
                'ongoing_actions': ongoing_actions,
                'emergency_contacts': emergency_contacts,
                'hazard_sections': hazard_sections,
            }
            
            print(f"Action plan generated with {len(immediate_actions)} immediate actions")
            return JsonResponse(action_plan)
            
        except Exception as e:
            print(f"Action plan error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
