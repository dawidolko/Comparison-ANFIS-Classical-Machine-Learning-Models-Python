#!/bin/bash
# Monitor pipeline and Streamlit progress

echo "🔍 PROJECT PROGRESS MONITOR"
echo "============================================================"
echo ""

while true; do
    clear
    echo "🔍 PROGRESS MONITOR - $(date '+%H:%M:%S')"
    echo "============================================================"
    echo ""
    
    # Check whether the setup.sh process is running
    if pgrep -f "setup.sh" > /dev/null; then
        echo "✅ Pipeline is running..."
        echo ""
        
        # Show the last 30 log lines
        echo "📊 CURRENT PROGRESS:"
        echo "------------------------------------------------------------"
        tail -30 setup_full.log | grep -E "(KROK|Epoch|Postęp|Train Accuracy|Test Accuracy|URUCHAMIANIE|✓|Trening)" || tail -30 setup_full.log
        echo "------------------------------------------------------------"
        
    elif pgrep -f "streamlit" > /dev/null; then
        echo "🎉 STREAMLIT RUNNING!"
        echo ""
        echo "🌐 Application available at:"
        echo "   👉 http://localhost:8501"
        echo ""
        echo "✅ Pipeline finished successfully!"
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        echo ""
        break
    else
        echo "⏸️  Pipeline stopped or finished"
        echo ""
        echo "📋 Last log lines:"
        echo "------------------------------------------------------------"
        tail -20 setup_full.log
        echo "------------------------------------------------------------"
        break
    fi
    
    echo ""
    echo "⏱️  Next update in 30 seconds..."
    echo "   (Press Ctrl+C to stop monitoring)"
    sleep 30
done
