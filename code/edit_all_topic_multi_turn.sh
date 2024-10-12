topics=(
    'art_sculpture' 'business_brand' 'business_industry' 'business_corporation' 
    'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    'geography_glacier' 'geography_volcano' 'geography_forest'
    'health_disease' 'health_symptom' 'health_medication'
    'technology_software' 'technology_programming_language' 'technology_database'
    'event_sport' 'event_history' 'event_film'
    'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    'places_country' 'places_city' 'places_landmark'
)

start_time=$(date +%s)

# If you have multiple GPUs, you can run experiments for multiple LLMs in parallel.
for topic in "${topics[@]}"; do
    python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=3 --topic_name="$topic" --model_name=llama2-7b --multi_turn=yes &
    python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=3 --topic_name="$topic" --model_name=llama3-8b --multi_turn=yes &
    python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=3 --topic_name="$topic" --model_name=mistral-7b --multi_turn=yes &
    wait
done

# Otherwise, run experiments for one LLM at a time.
# for topic in "${topics[@]}"; do
#     python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=3 --topic_name="$topic" --model_name=llama2-7b --multi_turn=yes
#     # python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=3 --topic_name="$topic" --model_name=llama3-8b --multi_turn=yes
#     # python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=3 --topic_name="$topic" --model_name=mistral-7b --multi_turn=yes
# done

end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
echo "Runtime for geography_forest: $runtime_minutes minutes"


