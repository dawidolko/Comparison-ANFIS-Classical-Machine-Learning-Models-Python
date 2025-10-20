#!/bin/bash
# Monitor postępu pipeline'u i Streamlit

echo "🔍 MONITOR POSTĘPU PROJEKTU"
echo "============================================================"
echo ""

while true; do
    clear
    echo "🔍 MONITOR POSTĘPU - $(date '+%H:%M:%S')"
    echo "============================================================"
    echo ""
    
    # Sprawdź czy proces setup.sh działa
    if pgrep -f "setup.sh" > /dev/null; then
        echo "✅ Pipeline działa..."
        echo ""
        
        # Pokaż ostatnie 30 linii logu
        echo "📊 AKTUALNY POSTĘP:"
        echo "------------------------------------------------------------"
        tail -30 setup_full.log | grep -E "(KROK|Epoch|Postęp|Train Accuracy|Test Accuracy|URUCHAMIANIE|✓|Trening)" || tail -30 setup_full.log
        echo "------------------------------------------------------------"
        
    elif pgrep -f "streamlit" > /dev/null; then
        echo "🎉 STREAMLIT URUCHOMIONY!"
        echo ""
        echo "🌐 Aplikacja dostępna pod adresem:"
        echo "   👉 http://localhost:8501"
        echo ""
        echo "✅ Pipeline zakończony pomyślnie!"
        echo ""
        echo "Naciśnij Ctrl+C aby zakończyć monitorowanie"
        echo ""
        break
    else
        echo "⏸️  Pipeline zatrzymany lub zakończony"
        echo ""
        echo "📋 Ostatnie linie logu:"
        echo "------------------------------------------------------------"
        tail -20 setup_full.log
        echo "------------------------------------------------------------"
        break
    fi
    
    echo ""
    echo "⏱️  Następna aktualizacja za 30 sekund..."
    echo "   (Naciśnij Ctrl+C aby przerwać monitorowanie)"
    sleep 30
done
