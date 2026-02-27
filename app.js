"use strict";
function showToast(msg, type = 'success') {
    const t = document.getElementById('toast');
    const ico = document.getElementById('toastIcon');
    t.className = `toast ${type}`;
    document.getElementById('toastMsg').textContent = msg;
    ico.className = { success: 'fas fa-check', error: 'fas fa-times', info: 'fas fa-info' }[type] || 'fas fa-check';
    t.classList.add('show');
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove('show'), 3200);
}
function calcSurplus() {
    const people = +document.getElementById('surPeople').value;
    const food = document.getElementById('surFood').value.trim();
    if (!people || !food) {
        showToast('Please fill all fields', 'error');
        return;
    }
    const base = window.location.origin;
    fetch(`${base}/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ food, people })
    })
        .then(r => r.json())
        .then(data => {
        const r = document.getElementById('surResult');
        r.style.display = 'block';
        if (data.error) {
            r.className = 'freshness-result bad';
            r.innerHTML = `⚠️ ${data.error}`;
            return;
        }
        r.className = 'freshness-result good';
        // Build aesthetic HTML
        let html = `
        <div style="padding: 20px; background: rgba(0,230,118,0.08); border-radius: 12px;">
          <div style="margin-bottom: 20px;">
            <h3 style="margin: 0 0 8px 0; color: #00e676; font-size: 24px; text-transform: capitalize;">
              ${data.food}
            </h3>
            <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 14px;">
              For <strong>${data.people_count} people</strong>
            </p>
          </div>
          
          <div style="background: rgba(0,230,118,0.15); padding: 16px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #00e676;">
            <p style="margin: 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Recommended Total Quantity</p>
            <p style="margin: 8px 0 0 0; font-size: 28px; font-weight: 700; color: #00e676;">
              ${data.recommended_total_quantity} <span style="font-size: 18px;">${data.unit}</span>
            </p>
          </div>
      `;
        // Ingredients table
        html += `
        <div style="margin-bottom: 20px;">
          <h4 style="margin: 0 0 12px 0; color: rgba(255,255,255,0.9); font-size: 16px;">📋 Ingredients Per Person</h4>
          <div style="background: rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden;">
            <table style="width: 100%; color: rgba(255,255,255,0.8); font-size: 13px;">
              <tbody>
      `;
        Object.entries(data.ingredients_per_person).forEach(([ing, val]) => {
            const displayVal = typeof val === 'number' ? val.toFixed(3) : val;
            html += `
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 10px 14px; text-transform: capitalize;">${ing.replace(/_/g, ' ')}</td>
            <td style="padding: 10px 14px; text-align: right; color: #00e676; font-weight: 600;">${displayVal}</td>
          </tr>
        `;
        });
        html += `
              </tbody>
            </table>
          </div>
        </div>
      `;
        // Total ingredients estimate
        html += `
        <div style="margin-bottom: 20px;">
          <h4 style="margin: 0 0 12px 0; color: rgba(255,255,255,0.9); font-size: 16px;">🛒 Total Ingredients Required</h4>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
      `;
        Object.entries(data.total_ingredients_estimate).forEach(([ing, item]) => {
            const qty = typeof item === 'object' ? item.quantity : item;
            const unit = typeof item === 'object' ? item.unit : '';
            html += `
          <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 8px; border: 1px solid rgba(0,230,118,0.2);">
            <p style="margin: 0 0 6px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: capitalize;">${ing.replace(/_/g, ' ')}</p>
            <p style="margin: 0; font-size: 16px; font-weight: 700; color: #00e676;">${qty} <span style="font-size: 12px;">${unit}</span></p>
          </div>
        `;
        });
        html += `
          </div>
        </div>
      `;
        // Nutrition info if available
        if (data.nutrition_estimate && !data.nutrition_estimate.message) {
            const n = data.nutrition_estimate;
            html += `
          <div style="margin-bottom: 20px;">
            <h4 style="margin: 0 0 12px 0; color: rgba(255,255,255,0.9); font-size: 16px;">🥗 Nutrition Information (Total)</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
              <div style="background: rgba(255,109,0,0.15); padding: 12px; border-radius: 8px; border-left: 3px solid #ff6d00;">
                <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.6); font-size: 11px; text-transform: uppercase;">Calories</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #ffb74d;">${n.total_calories_kcal} kcal</p>
              </div>
              <div style="background: rgba(244,67,30,0.15); padding: 12px; border-radius: 8px; border-left: 3px solid #f4431e;">
                <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.6); font-size: 11px; text-transform: uppercase;">Protein</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #ff6b6b;">${n.total_protein_g}g</p>
              </div>
              <div style="background: rgba(2,136,209,0.15); padding: 12px; border-radius: 8px; border-left: 3px solid #0288d1;">
                <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.6); font-size: 11px; text-transform: uppercase;">Carbs</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #4fc3f7;">${n.total_carbohydrates_g}g</p>
              </div>
              <div style="background: rgba(255,193,7,0.15); padding: 12px; border-radius: 8px; border-left: 3px solid #ffc107;">
                <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.6); font-size: 11px; text-transform: uppercase;">Fat</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #ffd54f;">${n.total_fat_g}g</p>
              </div>
            </div>
          </div>
        `;
        }
        html += `</div>`;
        r.innerHTML = html;
    })
        .catch(() => showToast('Error contacting server', 'error'));
}
function checkFreshness() {
    console.log('checkFreshness invoked');
    const food = document.getElementById('fFoodType').value.trim();
    const temp = +document.getElementById('fTemp').value;
    const hum = +document.getElementById('fHumidity').value;
    const prepInput = document.getElementById('fPrep').value;
    if (!food || !temp || !hum || !prepInput) {
        showToast('Please fill all required fields', 'error');
        return;
    }
    // keep the string as entered (YYYY-MM-DDTHH:MM) including the 'T'
    const prepared_time = prepInput;
    // Calculate hours ago for display based on input string
    const preparedDate = new Date(prepared_time);
    const hoursAgo = (Date.now() - preparedDate.getTime()) / (1000 * 60 * 60);
    const base = window.location.origin;
    fetch(`${base}/freshness`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ food, temp, humid: hum, prepared_time })
    })
        .then(r => r.json())
        .then(data => {
        const r = document.getElementById('freshResult');
        r.style.display = 'block';
        // Determine status styling and emoji
        let statusColor = '#00e676';
        let statusEmoji = '✅';
        let statusBg = 'rgba(0,230,118,0.15)';
        let statusBorder = '#00e676';
        if (data.status === 'High Risk') {
            statusColor = '#ffb74d';
            statusEmoji = '⚠️';
            statusBg = 'rgba(255,180,77,0.15)';
            statusBorder = '#ffb74d';
        }
        else if (data.status === 'Spoiled') {
            statusColor = '#ff6b6b';
            statusEmoji = '❌';
            statusBg = 'rgba(255,107,107,0.15)';
            statusBorder = '#ff6b6b';
        }
        r.className = 'freshness-result good';
        // Build beautiful HTML
        let html = `
        <div style="padding: 20px; background: rgba(0,230,118,0.08); border-radius: 12px;">
          <div style="margin-bottom: 20px;">
            <h3 style="margin: 0 0 8px 0; color: #00e676; font-size: 24px; text-transform: capitalize;">
              ${data.food}
            </h3>
            <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 14px;">
              Freshness Analysis
            </p>
          </div>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
            <!-- Status Card -->
            <div style="background: ${statusBg}; padding: 16px; border-radius: 8px; border-left: 4px solid ${statusBorder};">
              <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Status</p>
              <p style="margin: 0; font-size: 24px; font-weight: 700; color: ${statusColor};">
                ${statusEmoji} ${data.status}
              </p>
            </div>
            
            <!-- Remaining Hours Card -->
            <div style="background: rgba(66,165,245,0.15); padding: 16px; border-radius: 8px; border-left: 4px solid #42a5f5;">
              <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Time Remaining</p>
              <p style="margin: 0; font-size: 28px; font-weight: 700; color: #42a5f5;">
                ${data.remaining_hours.toFixed(1)}h
              </p>
            </div>
          </div>
          
          <!-- Freshness Bar -->
          <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 20px;">
            <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Freshness Level</p>
            <div style="width: 100%; height: 12px; background: rgba(255,255,255,0.1); border-radius: 6px; overflow: hidden;">
              <div style="height: 100%; background: linear-gradient(90deg, ${statusColor}, ${statusColor}); width: ${Math.min(100, (data.remaining_hours / 24) * 100)}%; transition: width 0.3s ease;"></div>
            </div>
          </div>
          
          <!-- Input Parameters -->
          <div style="margin-bottom: 20px;">
            <h4 style="margin: 0 0 12px 0; color: rgba(255,255,255,0.9); font-size: 16px;">🌡️ Analysis Parameters</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
              <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,180,77,0.3);">
                <p style="margin: 0 0 6px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase;">Temperature</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #ffb74d;">${temp}°C</p>
              </div>
              <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 8px; border: 1px solid rgba(100,200,255,0.3);">
                <p style="margin: 0 0 6px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase;">Humidity</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #4fc3f7;">${hum}%</p>
              </div>
              <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 8px; border: 1px solid rgba(0,230,118,0.3);">
                <p style="margin: 0 0 6px 0; color: rgba(255,255,255,0.6); font-size: 12px; text-transform: uppercase;">Prepared</p>
                <p style="margin: 0; font-size: 18px; font-weight: 700; color: #00e676;">${hoursAgo.toFixed(1)}h ago</p>
              </div>
            </div>
            <div style="margin-top: 6px; font-size: 12px; color: rgba(255,255,255,0.6);">
              Prepared at: <code style="background: rgba(255,255,255,0.1); padding:2px 4px; border-radius:4px;">${prepInput}</code>
            </div>
          </div>
          
          <!-- Status Message -->
          <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border-left: 3px solid ${statusBorder};">
      `;
        if (data.status === 'Safe') {
            html += `<p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 13px;">✅ Food is <strong>Safe to Consume</strong>. Current conditions allow for approximately <strong>${data.remaining_hours.toFixed(1)} more hours</strong> of freshness before spoilage risk.</p>`;
        }
        else if (data.status === 'High Risk') {
            html += `<p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 13px;">⚠️ Food has <strong>High Risk of Spoilage</strong>. Only <strong>${data.remaining_hours.toFixed(1)} hours</strong> remain. Consume or store immediately at lower temperature.</p>`;
        }
        else {
            html += `<p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 13px;">❌ Food appears to be <strong>Spoiled</strong>. Do not consume. Discard safely.</p>`;
        }
        html += `
          </div>
        </div>
      `;
        r.innerHTML = html;
    })
        .catch(() => showToast('Error contacting server', 'error'));
}
// Export to global window object for HTML to access
window.calcSurplus = calcSurplus;
window.checkFreshness = checkFreshness;
