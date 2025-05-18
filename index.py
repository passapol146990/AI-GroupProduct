from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ["เครื่องใช้ไฟฟ้า", "อาหารและเครื่องดื่ม", "รองเท้า", "เสื้อผ้า"]

sequence_to_classify = "นาฬิกา Garmin"

result = classifier(sequence_to_classify, candidate_labels)
print(result)
